import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


warnings.filterwarnings("ignore")


@dataclass
class ConfiguracionProyecto:
    """Configuracion general del experimento."""

    ruta_dataset: str = "salida_pcaps.csv"
    carpeta_salida: str = "resultados_abc_rf"

    semilla: int = 42
    test_size: float = 0.20
    validacion_size: float = 0.20

    poblacion_abc: int = 30
    iteraciones_abc: int = 15
    limite_abandono: int = 2
    paciencia_sin_mejora: int = 15

    muestra_optimizacion: int = 15000

    rango_n_estimators: Tuple[int, int] = (20, 80)
    rango_max_depth: Tuple[int, int] = (10, 25)
    rango_min_samples_split: Tuple[int, int] = (2, 10)
    rango_min_samples_leaf: Tuple[int, int] = (1, 5)
    rango_max_features: Tuple[float, float] = (0.03, 0.20)


@dataclass
class ResultadoExperimento:
    """Estructura para guardar el resultado final."""

    mejores_parametros: Dict[str, object]
    mejor_fitness: float
    iteracion_mejor_solucion: int
    tiempo_total: float
    metricas_prueba: Dict[str, float]
    matriz_confusion: np.ndarray
    clases: List[str]
    historial_fitness: List[float]
    historial_accuracy: List[float]
    historial_diversidad: List[float]
    historial_tiempo: List[float]


class CargadorDatos:
    """Carga, limpieza y preparacion del dataset."""

    def __init__(self, config: ConfiguracionProyecto):
        self.config = config
        self.columna_objetivo: Optional[str] = None
        self.nombres_clases: List[str] = []
        self.modo_clasificacion: str = "binario"
        self.encoder_objetivo: Optional[LabelEncoder] = None

    def cargar_dataset(self) -> pd.DataFrame:
        ruta = Path(self.config.ruta_dataset)
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontro el archivo: {ruta}")

        df = pd.read_csv(ruta, low_memory=False)
        if df.empty:
            raise ValueError("El dataset esta vacio.")

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.drop_duplicates()
        return df

    def detectar_columna_objetivo(self, df: pd.DataFrame) -> str:
        columnas = [c.lower().strip() for c in df.columns]

        if "type" in columnas:
            return df.columns[columnas.index("type")]
        if "label" in columnas:
            return df.columns[columnas.index("label")]

        raise ValueError("No se encontro una columna objetivo valida entre label o type.")

    def identificar_columnas_irrelevantes(self, df: pd.DataFrame, objetivo: str) -> List[str]:
        columnas_fijas = {
            "ts", "src_ip", "dst_ip", "http_uri", "http_referrer",
            "http_user_agent", "ssl_subject", "ssl_issuer"
        }

        columnas_a_eliminar = []
        for columna in df.columns:
            nombre = columna.lower().strip()
            if columna == objetivo:
                continue

            if nombre in columnas_fijas:
                columnas_a_eliminar.append(columna)
                continue

            if nombre in {"label", "type"} and columna != objetivo:
                columnas_a_eliminar.append(columna)
                continue

            if df[columna].dtype == "object":
                serie = df[columna].astype(str).fillna("")
                ratio_unicos = serie.nunique(dropna=True) / max(len(serie), 1)
                longitud_media = serie.str.len().mean()

                if ratio_unicos > 0.50 or longitud_media > 30:
                    columnas_a_eliminar.append(columna)

        return sorted(set(columnas_a_eliminar))

    def codificar_objetivo(self, serie: pd.Series) -> Tuple[np.ndarray, List[str], str]:
        """Convierte el objetivo a formato numerico y detecta clasificacion binaria o multiclase."""
        texto = serie.astype(str).str.strip().str.lower()

        palabras_normal = ["normal", "benign", "benigno", "legit", "legitimate", "good"]

        hay_normal = texto.apply(lambda x: any(p in x for p in palabras_normal)).any()

        if hay_normal:
            y = texto.apply(lambda x: 0 if any(p in x for p in palabras_normal) else 1).astype(int).values
            return y, ["normal", "ataque"], "binario"

        if pd.api.types.is_numeric_dtype(serie):
            valores = serie.dropna().unique()
            if len(valores) == 2:
                valores_ordenados = sorted(valores)
                mapeo = {valores_ordenados[0]: 0, valores_ordenados[1]: 1}
                y = serie.map(mapeo).astype(int).values
                return y, [str(valores_ordenados[0]), str(valores_ordenados[1])], "binario"

        le = LabelEncoder()
        y = le.fit_transform(serie.astype(str))
        clases = list(le.classes_)
        modo = "binario" if len(clases) == 2 else "multiclase"
        return y, clases, modo

    def crear_preprocesador(self, X: pd.DataFrame) -> ColumnTransformer:
        """Crea el preprocesador para variables numericas y categoricas."""
        columnas_numericas = X.select_dtypes(include=[np.number]).columns.tolist()
        columnas_categoricas = [c for c in X.columns if c not in columnas_numericas]

        transformadores = []

        if columnas_numericas:
            transformadores.append(
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputador", SimpleImputer(strategy="median"))
                        ]
                    ),
                    columnas_numericas,
                )
            )

        if columnas_categoricas:
            try:
                codificador = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                codificador = OneHotEncoder(handle_unknown="ignore", sparse=False)

            transformadores.append(
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputador", SimpleImputer(strategy="most_frequent")),
                            ("codificador", codificador),
                        ]
                    ),
                    columnas_categoricas,
                )
            )

        if not transformadores:
            raise ValueError("No se encontraron columnas utilizables para el entrenamiento.")

        preprocesador = ColumnTransformer(
            transformers=transformadores,
            remainder="drop",
            sparse_threshold=0,
        )
        return preprocesador

    def preparar_datos(self):
        """Carga, limpia, codifica y divide los datos."""
        print("Cargando dataset")
        df = self.cargar_dataset()

        self.columna_objetivo = self.detectar_columna_objetivo(df)
        print(f"Columna objetivo detectada: {self.columna_objetivo}")

        columnas_irrelevantes = self.identificar_columnas_irrelevantes(df, self.columna_objetivo)
        if columnas_irrelevantes:
            print("Columnas ignoradas por irrelevancia o exceso de texto:")
            for col in columnas_irrelevantes:
                print(f"  - {col}")

        df = df.drop(columns=columnas_irrelevantes, errors="ignore")
        df = df.dropna(subset=[self.columna_objetivo])

        y, clases, modo = self.codificar_objetivo(df[self.columna_objetivo])
        self.nombres_clases = clases
        self.modo_clasificacion = modo

        columnas_a_eliminar_por_fuga = [self.columna_objetivo]
        for col in ["label", "type"]:
            if col in df.columns and col != self.columna_objetivo:
                columnas_a_eliminar_por_fuga.append(col)

        X = df.drop(columns=columnas_a_eliminar_por_fuga, errors="ignore")
        if X.empty:
            raise ValueError("No quedaron columnas de entrada despues del filtrado.")

        preprocesador = self.crear_preprocesador(X)

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.semilla,
            stratify=y,
        )

        X_train = preprocesador.fit_transform(X_train_raw)
        X_test = preprocesador.transform(X_test_raw)

        X_train = np.asarray(X_train, dtype=float)
        X_test = np.asarray(X_test, dtype=float)
        y_train = np.asarray(y_train, dtype=int)
        y_test = np.asarray(y_test, dtype=int)

        print("Preprocesamiento completado.")
        print(f"Forma entrenamiento: {X_train.shape}")
        print(f"Forma prueba: {X_test.shape}")
        print(f"Modo de clasificacion: {self.modo_clasificacion}")
        print(f"Clases detectadas: {self.nombres_clases}")

        return X_train, X_test, y_train, y_test, X.columns.tolist()


class EvaluadorModelo:
    """Calcula las metricas principales del modelo."""

    def __init__(self, modo_clasificacion: str):
        self.modo_clasificacion = modo_clasificacion

    def calcular_fpr(self, y_real: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula la tasa de falsos positivos."""
        etiquetas = np.unique(np.concatenate([y_real, y_pred]))
        cm = confusion_matrix(y_real, y_pred, labels=etiquetas)

        if len(etiquetas) == 2:
            if cm.shape == (1, 1):
                return 0.0
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                return fp / (fp + tn) if (fp + tn) > 0 else 0.0

        fprs = []
        for i in range(len(etiquetas)):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            valor = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fprs.append(valor)

        return float(np.mean(fprs)) if fprs else 0.0

    def evaluar(self, modelo, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Obtiene accuracy, recall y FPR."""
        y_pred = modelo.predict(X)
        accuracy = accuracy_score(y, y_pred)

        if self.modo_clasificacion == "binario":
            recall = recall_score(y, y_pred, zero_division=0)
        else:
            recall = recall_score(y, y_pred, average="macro", zero_division=0)

        fpr = self.calcular_fpr(y, y_pred)

        return {
            "accuracy": float(accuracy),
            "recall": float(recall),
            "fpr": float(fpr),
        }


class OptimizadorABC:
    """Artificial Bee Colony para optimizar hiperparametros de Random Forest."""

    def __init__(
        self,
        config,
        X_train: np.ndarray,
        y_train: np.ndarray,
        evaluador,
    ):
        self.config = config
        self.evaluador = evaluador
        self.rng = np.random.default_rng(config.semilla)

        self.X_train_completo = X_train
        self.y_train_completo = y_train

        self.X_abc, self.y_abc = self._tomar_muestra_estratificada(
            X_train,
            y_train,
            config.muestra_optimizacion,
        )

        self.X_subtrain, self.X_validacion, self.y_subtrain, self.y_validacion = (
            train_test_split(
                self.X_abc,
                self.y_abc,
                test_size=config.validacion_size,
                random_state=config.semilla,
                stratify=self.y_abc,
            )
        )

        self.limites = np.array(
            [
                [config.rango_n_estimators[0], config.rango_n_estimators[1]],
                [config.rango_max_depth[0], config.rango_max_depth[1]],
                [
                    config.rango_min_samples_split[0],
                    config.rango_min_samples_split[1],
                ],
                [config.rango_min_samples_leaf[0], config.rango_min_samples_leaf[1]],
                [config.rango_max_features[0], config.rango_max_features[1]],
            ],
            dtype=float,
        )

        self.evaluaciones_cache: Dict[
            Tuple[float, ...], Tuple[float, Dict[str, float]]
        ] = {}

        self.mejor_solucion: Optional[np.ndarray] = None
        self.mejor_fitness: float = -np.inf
        self.mejor_metricas: Dict[str, float] = {}
        self.mejor_iteracion: int = 0

        self.historial_fitness: List[float] = []
        self.historial_accuracy: List[float] = []
        self.historial_diversidad: List[float] = []
        self.historial_tiempo: List[float] = []

    def _tomar_muestra_estratificada(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_muestras: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Toma una muestra pequena para acelerar ABC sin perder estratificacion."""
        if len(X) <= max_muestras:
            return X, y

        X_muestra, _, y_muestra, _ = train_test_split(
            X,
            y,
            train_size=max_muestras,
            random_state=self.config.semilla,
            stratify=y,
        )
        return X_muestra, y_muestra

    def generar_solucion(self) -> np.ndarray:
        sol = np.array(
            [
                self.rng.integers(self.limites[0, 0], self.limites[0, 1] + 1),
                self.rng.integers(self.limites[1, 0], self.limites[1, 1] + 1),
                self.rng.integers(self.limites[2, 0], self.limites[2, 1] + 1),
                self.rng.integers(self.limites[3, 0], self.limites[3, 1] + 1),
                self.rng.uniform(self.limites[4, 0], self.limites[4, 1]),
            ],
            dtype=float,
        )
        return self.ajustar_solucion(sol)

    def ajustar_solucion(self, sol: np.ndarray) -> np.ndarray:
        sol = sol.copy()
        sol[0] = int(np.clip(round(sol[0]), self.limites[0, 0], self.limites[0, 1]))
        sol[1] = int(np.clip(round(sol[1]), self.limites[1, 0], self.limites[1, 1]))
        sol[2] = int(np.clip(round(sol[2]), self.limites[2, 0], self.limites[2, 1]))
        sol[3] = int(np.clip(round(sol[3]), self.limites[3, 0], self.limites[3, 1]))
        sol[4] = float(np.clip(sol[4], self.limites[4, 0], self.limites[4, 1]))
        return sol

    def decodificar_solucion(self, sol: np.ndarray) -> Dict[str, object]:
        """Convierte un vector en hiperparametros de Random Forest."""
        return {
            "n_estimators": int(sol[0]),
            "max_depth": int(sol[1]),
            "min_samples_split": int(sol[2]),
            "min_samples_leaf": int(sol[3]),
            "max_features": float(sol[4]),
        }

    def evaluar_solucion(self, sol: np.ndarray):
        clave = tuple(np.round(sol, 6))
        if clave in self.evaluaciones_cache:
            return self.evaluaciones_cache[clave]

        if len(self.evaluaciones_cache) > 1000:
            self.evaluaciones_cache.pop(next(iter(self.evaluaciones_cache)))

        params = self.decodificar_solucion(sol)

        # Detectamos automáticamente las etiquetas únicas en y_subtrain
        clases_unicas = np.unique(self.y_subtrain)
        pesos_personalizados = {}
        for c in clases_unicas:
            if str(c).lower().strip() == 'normal' or c == 0:
                pesos_personalizados[c] = 7.5
            else:
                pesos_personalizados[c] = 1.0

        modelo = RandomForestClassifier(
            **params,
            class_weight=pesos_personalizados,
            random_state=self.config.semilla,
            n_jobs=-1,
            bootstrap=True,
        )

        modelo.fit(self.X_subtrain, self.y_subtrain)
        metricas = self.evaluador.evaluar(modelo, self.X_validacion, self.y_validacion)

        recall = metricas["recall"]
        fpr = metricas["fpr"]
        especificidad = max(0.0, 1.0 - fpr)
        if recall + especificidad == 0:
            fitness = 0.0
        else:
            beta = 2.0
            fitness = ((1 + beta ** 2) * (especificidad * recall)) / ((beta ** 2 * especificidad) + recall)

        # Penalización si el FPR supera límites operativos aceptables
        if fpr >= 0.20:
            fitness *= 0.5

        resultado = (float(fitness), metricas)
        self.evaluaciones_cache[clave] = resultado
        return resultado

    def calcular_diversidad(self, poblacion: np.ndarray) -> float:
        """Mide la diversidad promedio de la poblacion."""
        poblacion_norm = poblacion.astype(float).copy()

        for i in range(poblacion_norm.shape[1]):
            minimo = self.limites[i, 0]
            maximo = self.limites[i, 1]
            if maximo > minimo:
                poblacion_norm[:, i] = (poblacion_norm[:, i] - minimo) / (
                    maximo - minimo
                )
            else:
                poblacion_norm[:, i] = 0.0

        return float(np.mean(np.std(poblacion_norm, axis=0)))

    def mover_solucion(
        self, sol_actual: np.ndarray, sol_vecina: np.ndarray
    ) -> np.ndarray:
        """Aplica la regla de busqueda local del ABC."""
        nueva = sol_actual.copy()
        dimension = self.rng.integers(0, len(sol_actual))
        phi = self.rng.uniform(-1, 1)
        nueva[dimension] = sol_actual[dimension] + phi * (
            sol_actual[dimension] - sol_vecina[dimension]
        )
        return self.ajustar_solucion(nueva)

    def optimizar(self):
        """Ejecuta ABC con parada temprana para reducir tiempo."""
        print("Iniciando optimizacion ABC")
        inicio_total = time.time()

        poblacion = np.array(
            [self.generar_solucion() for _ in range(self.config.poblacion_abc)]
        )
        fitness = np.zeros(self.config.poblacion_abc, dtype=float)
        metricas_poblacion: List[Dict[str, float]] = [
            None
        ] * self.config.poblacion_abc  # type: ignore
        contadores = np.zeros(self.config.poblacion_abc, dtype=int)

        for i in range(self.config.poblacion_abc):
            fitness[i], metricas_poblacion[i] = self.evaluar_solucion(poblacion[i])

        idx_mejor = int(np.argmax(fitness))
        self.mejor_solucion = poblacion[idx_mejor].copy()
        self.mejor_fitness = float(fitness[idx_mejor])
        self.mejor_metricas = metricas_poblacion[idx_mejor].copy()
        self.mejor_iteracion = 0

        sin_mejora = 0

        for iteracion in range(self.config.iteraciones_abc):
            mejora_iteracion = False

            # FASES DE ABEJAS EMPLEADAS
            for i in range(self.config.poblacion_abc):
                candidatos = list(range(self.config.poblacion_abc))
                candidatos.remove(i)
                k = int(self.rng.choice(candidatos))

                nueva_solucion = self.mover_solucion(poblacion[i], poblacion[k])
                nuevo_fitness, nuevas_metricas = self.evaluar_solucion(nueva_solucion)

                if nuevo_fitness > fitness[i]:
                    poblacion[i] = nueva_solucion
                    fitness[i] = nuevo_fitness
                    metricas_poblacion[i] = nuevas_metricas
                    contadores[i] = 0
                else:
                    contadores[i] += 1

            # SELECCION DE ABEJAS OBSERVADORAS
            pesos = fitness - fitness.min() + 1e-9
            probabilidades = pesos / pesos.sum()

            for _ in range(self.config.poblacion_abc):
                i = int(self.rng.choice(self.config.poblacion_abc, p=probabilidades))
                candidatos = list(range(self.config.poblacion_abc))
                candidatos.remove(i)
                k = int(self.rng.choice(candidatos))

                nueva_solucion = self.mover_solucion(poblacion[i], poblacion[k])
                nuevo_fitness, nuevas_metricas = self.evaluar_solucion(nueva_solucion)

                if nuevo_fitness > fitness[i]:
                    poblacion[i] = nueva_solucion
                    fitness[i] = nuevo_fitness
                    metricas_poblacion[i] = nuevas_metricas
                    contadores[i] = 0
                else:
                    contadores[i] += 1

            # FASE DE ABEJAS EXPLORADORAS
            for i in range(self.config.poblacion_abc):
                if contadores[i] >= self.config.limite_abandono:
                    poblacion[i] = self.generar_solucion()
                    fitness[i], metricas_poblacion[i] = self.evaluar_solucion(
                        poblacion[i]
                    )
                    contadores[i] = 0

            # EVALUACION DE LA ITERACION
            idx_mejor_iter = int(np.argmax(fitness))
            mejor_fitness_iter = float(fitness[idx_mejor_iter])
            mejor_metricas_iter = metricas_poblacion[idx_mejor_iter]

            if mejor_fitness_iter > self.mejor_fitness:
                self.mejor_fitness = mejor_fitness_iter
                self.mejor_solucion = poblacion[idx_mejor_iter].copy()
                self.mejor_metricas = mejor_metricas_iter.copy()
                self.mejor_iteracion = iteracion + 1
                mejora_iteracion = True
                sin_mejora = 0
            else:
                sin_mejora += 1

            diversidad = self.calcular_diversidad(poblacion)
            tiempo_acumulado = time.time() - inicio_total

            self.historial_fitness.append(mejor_fitness_iter)
            self.historial_accuracy.append(mejor_metricas_iter["accuracy"])
            self.historial_diversidad.append(diversidad)
            self.historial_tiempo.append(tiempo_acumulado)

            print(
                f"Iteracion {iteracion + 1}/{self.config.iteraciones_abc} | "
                f"Fitness: {mejor_fitness_iter:.4f} | "
                f"Accuracy: {mejor_metricas_iter['accuracy']:.4f} | "
                f"Recall: {mejor_metricas_iter['recall']:.4f} | "
                f"FPR: {mejor_metricas_iter['fpr']:.4f} | "
                f"Diversidad: {diversidad:.4f} | "
                f"Tiempo: {tiempo_acumulado:.2f}s"
            )

            if (
                not mejora_iteracion
                and sin_mejora >= self.config.paciencia_sin_mejora
            ):
                print(
                    "No hubo mejora reciente. Se activa parada temprana para ahorrar tiempo."
                )
                break

        tiempo_total = time.time() - inicio_total

        return {
            "mejor_solucion": self.mejor_solucion,
            "mejor_fitness": self.mejor_fitness,
            "mejor_metricas_validacion": self.mejor_metricas,
            "iteracion_mejor_solucion": self.mejor_iteracion,
            "tiempo_total": tiempo_total,
            "historial_fitness": self.historial_fitness,
            "historial_accuracy": self.historial_accuracy,
            "historial_diversidad": self.historial_diversidad,
            "historial_tiempo": self.historial_tiempo,
        }


class GeneradorReportes:
    """Genera graficas, matriz de confusion y archivo TXT."""

    def __init__(self, config: ConfiguracionProyecto):
        self.config = config
        self.carpeta = Path(config.carpeta_salida)
        self.carpeta.mkdir(parents=True, exist_ok=True)

    def guardar_linea(self, x, y, titulo, etiqueta_y, nombre_archivo):
        """Guarda una grafica de linea."""
        plt.figure(figsize=(9, 5))
        plt.plot(x, y)
        plt.title(titulo)
        plt.xlabel("Iteracion")
        plt.ylabel(etiqueta_y)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.carpeta / nombre_archivo, dpi=200)
        plt.close()

    def guardar_tiempo(self, tiempos):
        """Guarda la evolucion del tiempo de ejecucion."""
        plt.figure(figsize=(9, 5))
        plt.plot(range(1, len(tiempos) + 1), tiempos)
        plt.title("Evolucion del tiempo de ejecucion")
        plt.xlabel("Iteracion")
        plt.ylabel("Tiempo acumulado (s)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.carpeta / "tiempo_ejecucion.png", dpi=200)
        plt.close()

    def guardar_matriz_confusion(self, cm: np.ndarray, clases: List[str]):
        """Guarda la matriz de confusion en PNG."""
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Matriz de confusion")
        plt.colorbar()
        ticks = np.arange(len(clases))
        plt.xticks(ticks, clases, rotation=45, ha="right")
        plt.yticks(ticks, clases)
        plt.ylabel("Real")
        plt.xlabel("Prediccion")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")

        plt.tight_layout()
        plt.savefig(self.carpeta / "matriz_confusion.png", dpi=200)
        plt.close()

    def guardar_txt(
        self,
        resultado: ResultadoExperimento,
        nombre_dataset: str,
        columna_objetivo: str,
        modo_clasificacion: str,
    ) -> Path:
        """Escribe un reporte academico en TXT."""
        archivo = self.carpeta / "resultados_abc_rf.txt"

        lineas = []
        lineas.append("REPORTE DEL EXPERIMENTO\n")
        lineas.append("=" * 80 + "\n")
        lineas.append("TITULO TENTATIVO:\n")
        lineas.append(
            "Optimizacion bio-inspirada para el ajuste de hiperparametros para la precision "
            "en los modelos de deteccion dentro de las redes IoT implementadas en industria 4.0, 2026.\n"
        )
        lineas.append("\nVARIABLE INDEPENDIENTE:\n")
        lineas.append("Optimizacion bio-inspirada para el ajuste de hiperparametros\n")
        lineas.append("\nVARIABLE DEPENDIENTE:\n")
        lineas.append("Precision en los modelos de deteccion\n")
        lineas.append("\nOBJETIVO DETECTADO:\n")
        lineas.append(f"{columna_objetivo}\n")
        lineas.append("\nMODO DE CLASIFICACION:\n")
        lineas.append(f"{modo_clasificacion}\n")
        lineas.append("\nDATASET:\n")
        lineas.append(f"{nombre_dataset}\n")
        lineas.append("\nCLASES DETECTADAS:\n")
        lineas.append(", ".join(resultado.clases) + "\n")

        lineas.append("\nMEJORES HIPERPARAMETROS ENCONTRADOS:\n")
        for k, v in resultado.mejores_parametros.items():
            lineas.append(f"- {k}: {v}\n")

        lineas.append("\nRESULTADOS DE ABC:\n")
        lineas.append(f"Mejor fitness: {resultado.mejor_fitness:.6f}\n")
        lineas.append(f"Iteracion de mejor solucion: {resultado.iteracion_mejor_solucion}\n")
        lineas.append(f"Tiempo total de optimizacion: {resultado.tiempo_total:.4f} s\n")

        lineas.append("\nMETRICAS SOBRE EL CONJUNTO DE PRUEBA:\n")
        lineas.append(f"Accuracy: {resultado.metricas_prueba['accuracy']:.6f}\n")
        lineas.append(f"Recall: {resultado.metricas_prueba['recall']:.6f}\n")
        lineas.append(f"FPR: {resultado.metricas_prueba['fpr']:.6f}\n")

        lineas.append("\nMATRIZ DE CONFUSION:\n")
        for fila in resultado.matriz_confusion:
            lineas.append(" ".join(str(int(x)) for x in fila) + "\n")

        lineas.append("\nINTERPRETACION:\n")
        lineas.append(
            "ABC mejora el ajuste de Random Forest porque no usa una unica configuracion fija, "
            "sino que explora varias soluciones candidatas y conserva las mas prometedoras. "
            "La fase de abejas empleadas explora alrededor de cada solucion, la fase de observadoras "
            "prioriza las mejores, y la fase exploradora evita que el proceso se estanque en una mala region "
            "del espacio de busqueda. Esto permite encontrar hiperparametros con mejor equilibrio entre "
            "accuracy, recall y FPR.\n"
        )
        lineas.append(
            "La diversidad de poblacion indica si el algoritmo aun esta explorando soluciones distintas. "
            "El tiempo total representa el costo computacional del ajuste automatico. En un contexto de "
            "investigacion, este enfoque justifica que la optimizacion bio-inspirada puede mejorar el "
            "rendimiento del clasificador y hacer mas solida la deteccion de trafico malicioso en redes IoT.\n"
        )

        with open(archivo, "w", encoding="utf-8") as f:
            f.writelines(lineas)

        return archivo

    def guardar_todo(
        self,
        resultado: ResultadoExperimento,
        nombre_dataset: str,
        columna_objetivo: str,
        modo_clasificacion: str,
    ) -> Path:
        """Genera todas las salidas del experimento."""
        self.guardar_linea(
            range(1, len(resultado.historial_fitness) + 1),
            resultado.historial_fitness,
            "Evolucion del fitness",
            "Fitness",
            "fitness_abc.png",
        )

        self.guardar_linea(
            range(1, len(resultado.historial_accuracy) + 1),
            resultado.historial_accuracy,
            "Evolucion de accuracy",
            "Accuracy",
            "accuracy_abc.png",
        )

        self.guardar_tiempo(resultado.historial_tiempo)

        self.guardar_linea(
            range(1, len(resultado.historial_diversidad) + 1),
            resultado.historial_diversidad,
            "Diversidad de la poblacion",
            "Diversidad",
            "diversidad_poblacion.png",
        )

        self.guardar_matriz_confusion(resultado.matriz_confusion, resultado.clases)

        return self.guardar_txt(
            resultado=resultado,
            nombre_dataset=nombre_dataset,
            columna_objetivo=columna_objetivo,
            modo_clasificacion=modo_clasificacion,
        )


class ProyectoABC_RandomForest:
    """Orquesta todo el flujo del experimento."""

    def __init__(self):
        self.config = ConfiguracionProyecto()
        self.cargador = CargadorDatos(self.config)
        self.reportes = GeneradorReportes(self.config)

    def ejecutar(self):
        try:
            print("\n" + "=" * 80)
            print("PROYECTO DE OPTIMIZACION ABC + RANDOM FOREST")
            print("=" * 80 + "\n")

            X_train, X_test, y_train, y_test, columnas_utilizadas = self.cargador.preparar_datos()

            evaluador = EvaluadorModelo(self.cargador.modo_clasificacion)
            optimizador = OptimizadorABC(
                config=self.config,
                X_train=X_train,
                y_train=y_train,
                evaluador=evaluador,
            )

            # FASE 1: OPTIMIZACION
            resultado_abc = optimizador.optimizar()

            mejores_parametros = optimizador.decodificar_solucion(resultado_abc["mejor_solucion"])

            # FASE 2: CONFIGURACION DE PESOS ASIMETRICOS
            # Esto asegura que el modelo final castigue el error en tráfico 'normal'
            # igual que lo hizo el optimizador ABC.
            clases_unicas = np.unique(y_train)
            pesos_finales = {}
            for c in clases_unicas:
                nombre_c = str(c).lower().strip()
                if nombre_c == 'normal' or c == 0:
                    pesos_finales[c] = 7.5
                else:
                    pesos_finales[c] = 1.0

            print("\nEntrenando modelo final con pesos asimétricos y mejores hiperparametros")

            # FASE 3: ENTRENAMIENTO FINAL
            modelo_final = RandomForestClassifier(
                **mejores_parametros,
                class_weight=pesos_finales,
                random_state=self.config.semilla,
                n_jobs=-1,
                bootstrap=True
            )
            modelo_final.fit(X_train, y_train)

            # FASE 4: EVALUACION Y PREDICCION
            y_pred = modelo_final.predict(X_test)

            metricas_prueba = evaluador.evaluar(modelo_final, X_test, y_test)

            etiquetas = np.unique(np.concatenate([y_test, y_pred]))
            cm = confusion_matrix(y_test, y_pred, labels=etiquetas)

            resultado = ResultadoExperimento(
                mejores_parametros=mejores_parametros,
                mejor_fitness=resultado_abc["mejor_fitness"],
                iteracion_mejor_solucion=resultado_abc["iteracion_mejor_solucion"],
                tiempo_total=resultado_abc["tiempo_total"],
                metricas_prueba=metricas_prueba,
                matriz_confusion=cm,
                clases=self.cargador.nombres_clases,
                historial_fitness=resultado_abc["historial_fitness"],
                historial_accuracy=resultado_abc["historial_accuracy"],
                historial_diversidad=resultado_abc["historial_diversidad"],
                historial_tiempo=resultado_abc["historial_tiempo"],
            )

            # FASE 5: REPORTE
            archivo_txt = self.reportes.guardar_todo(
                resultado=resultado,
                nombre_dataset=self.config.ruta_dataset,
                columna_objetivo=self.cargador.columna_objetivo or "desconocida",
                modo_clasificacion=self.cargador.modo_clasificacion,
            )

            print("\n" + "=" * 80)
            print("RESULTADOS FINALES (OPTIMIZADOS PARA OPERACIÓN REAL)")
            print("=" * 80)
            print(f"Mejor fitness encontrado: {resultado.mejor_fitness:.6f}")
            print(f"Iteración de mejor solución: {resultado.iteracion_mejor_solucion}")
            print(f"Tiempo total de optimización: {resultado.tiempo_total:.4f} s")

            print("\nMejores hiperparametros encontrados:")
            for clave, valor in resultado.mejores_parametros.items():
                print(f"  {clave}: {valor}")

            print("\nMetricas sobre el conjunto de prueba:")
            print(f"  Accuracy: {resultado.metricas_prueba['accuracy']:.6f}")
            print(f"  Recall:   {resultado.metricas_prueba['recall']:.6f}")
            print(f"  FPR:      {resultado.metricas_prueba['fpr']:.6f}")

            print("\nMatriz de confusion:")
            print(resultado.matriz_confusion)

            print("\nArchivos generados:")
            print(f"  {archivo_txt}")

        except FileNotFoundError as e:
            print(f"Error de archivo: {e}")
        except ValueError as e:
            print(f"Error de validación: {e}")
        except Exception as e:
            print(f"Error inesperado: {e}")


class Proyecto_RandomForest_Base:

    def __init__(self):
        self.config = ConfiguracionProyecto()
        self.cargador = CargadorDatos(self.config)

    def ejecutar(self):
        try:
            print("\n" + "=" * 80)
            print("PROYECTO LINEA BASE: RANDOM FOREST ESTÁNDAR (SIN OPTIMIZAR)")
            print("=" * 80 + "\n")

            # 1. Cargar los mismos datos (las mismas 27 columnas)
            X_train, X_test, y_train, y_test, columnas_utilizadas = self.cargador.preparar_datos()
            from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

            print("Entrenando Random Forest con hiperparámetros por defecto\n")
            inicio = time.time()
            modelo_base = RandomForestClassifier(
                random_state=self.config.semilla,
                n_jobs=-1
            )

            modelo_base.fit(X_train, y_train)
            tiempo_entrenamiento = time.time() - inicio
            y_pred = modelo_base.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            vn, fp, fn, vp = cm.ravel()

            accuracy = (vp + vn) / (vp + vn + fp + fn)
            recall = vp / (vp + fn) if (vp + fn) > 0 else 0
            fpr = fp / (fp + vn) if (fp + vn) > 0 else 0

            print("=" * 80)
            print("RESULTADOS DEL MODELO BASE (SIN OPTIMIZACIÓN)")
            print("=" * 80)
            print(f"Tiempo de entrenamiento: {tiempo_entrenamiento:.4f} s")
            print("\nMétricas sobre el conjunto de prueba:")
            print(f"  Accuracy: {accuracy:.6f}")
            print(f"  Recall:   {recall:.6f}")
            print(f"  FPR:      {fpr:.6f}")

            print("\nMatriz de confusión:")
            print(cm)

        except Exception as e:
            print(f"Error inesperado: {e}")

if __name__ == "__main__":
    proyecto = ProyectoABC_RandomForest()
    proyecto.ejecutar()
    experimento_base = Proyecto_RandomForest_Base()
    experimento_base.ejecutar()