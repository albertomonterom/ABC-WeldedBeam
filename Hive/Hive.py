# ---- IMPORTAR MÓDULOS

import numpy as np
import random
import sys
import copy


# ---- CLASE ABEJA

class Bee(object):
    """ Crea un objeto abeja. """

    def __init__(self, lower, upper, fun, funcon=None):
        """
        Instancia un objeto de abeja de manera aleatoria.

        Parámetros:
        ----------
            :param list lower  : límite inferior del vector solución
            :param list upper  : límite superior del vector solución
            :param def  fun    : función de evaluación
            :param def  funcon : función de restricciones unificada
        """

        # crea un vector de solución aleatorio
        self._random(lower, upper)

        # evalúa restricciones y función objetivo
        self._evaluate_solution(fun, funcon)

        # inicializa el contador de intentos fallidos (abandono)
        self.counter = 0

    def _random(self, lower, upper):
        """ Inicializa un vector solución de manera aleatoria. """
        self.vector = []
        for i in range(len(lower)):
            self.vector.append(lower[i] + random.random() * (upper[i] - lower[i]))

    def _evaluate_solution(self, fun, funcon):
        """Evalúa la solución: función objetivo y restricciones."""
        
        # Evalúa función objetivo
        if fun is not None:
            self.value = fun(self.vector)
        else:
            self.value = sys.float_info.max

        # Evalúa restricciones usando la función unificada
        if funcon is not None:
            constraint_result = funcon(self.vector)
            self.valid = constraint_result['feasible']
            self.violation = constraint_result['violation']
        else:
            self.valid = True
            self.violation = 0.0

        # Calcula fitness
        self._fitness()

    def _fitness(self):
        """
        Evalúa la aptitud (fitness) de un vector solución.
        La aptitud es una medida de calidad de una solución.
        """
        if self.value >= 0:
            self.fitness = 1 / (1 + self.value)
        else:
            self.fitness = 1 + abs(self.value)

class BeeHive(object):
    """
    Crea una Colonia Artificial de Abejas (ABC por sus siglas en inglés).

    La población de la colmena se compone de tres tipos distintos de individuos:
        1. "empleadas" (employees),
        2. "observadoras" (onlookers),
        3. "exploradoras" (scouts).

    Las abejas empleadas y las observadoras explotan las fuentes de néctar
    alrededor de la colmena, es decir, la fase de explotación, mientras que
    las exploradoras buscan nuevas soluciones en el dominio (fase de exploración).
    """

    def run(self):
        """ Ejecuta el algoritmo de Colonia Artificial de Abejas (ABC). """

        cost = {}
        cost["best"] = []
        cost["mean"] = []
        
        for itr in range(self.max_itrs):
            # fase de abejas empleadas
            for index in range(self.size):
                self.send_employee(index)

            # fase de abejas observadoras
            self.send_onlookers()

            # fase de abejas exploradoras
            self.send_scout()

            # calcula el mejor camino (mejor solución encontrada)
            self.find_best()

            # guarda la información de convergencia
            cost["best"].append(self.best)
            cost["mean"].append(sum([bee.value for bee in self.population]) / self.size)

            # imprime información del proceso si verbose está activado
            if self.verbose:
                self._verbose(itr, cost)
                if self.solution:
                    print(f"  Mejor vector x: {self.solution}")

        return cost

    def __init__(self, lower, upper, fun=None, numb_bees=30, max_itrs=100,
                 limit=None, modification_rate=1.0, selfun=None, seed=None, 
                 verbose=False, extra_params=None, funcon=None, p_f=0.45):
        """
        Instancia un objeto colmena.

        1. FASE DE INICIALIZACIÓN.
        -----------------------

        La población inicial de abejas debe cubrir todo el espacio de búsqueda
        lo más posible, generando individuos aleatoriamente dentro de los límites
        inferiores y superiores especificados.

        Parámetros:
        ----------
            :param list lower          : límite inferior del vector solución
            :param list upper          : límite superior del vector solución
            :param def fun             : función de evaluación del problema de optimización
            :param def funcon          : función de restricciones unificada (devuelve dict)
            :param float p_f           : probabilidad de comparar por función objetivo durante la jerarquización estocástica (0 ≤ p_f ≤ 1)
            :param int numb_bees       : número de abejas activas en la columna
            :param int limit           : límite de abandono de fuente (número máximo de intentos sin mejora antes de abandonar)
            :param float modification_rate : tasa o factor de modificación de la solución (controla el tamaño del paso en la mutación)
            :param def selfun          : función personalizada de selección
            :param int seed            : semilla del generador de números aleatorios
            :param boolean verbose     : muestra información detallada durante la ejecución
            :param dict extra_params   : argumentos opcionales para la función de selección `selfun`
        """

        # valida que los límites tengan la misma dimensión
        assert len(upper) == len(lower), "'lower' and 'upper' deben tener la misma longitud."

        # genera una semilla aleatoria si no se proporciona
        if seed is None:
            self.seed = random.randint(0, 1000)
        else:
            self.seed = seed
        random.seed(self.seed)

        # calcula el número de abejas empleadas (siempre par)
        self.size = int((numb_bees + numb_bees % 2))

        # asigna propiedades del algoritmo
        self.dim = len(lower)
        self.max_itrs = max_itrs
        
        # PARÁMETRO D: limit - Límite de abandono de fuente
        if limit is None:
            self.limit = int(0.6 * self.size * self.dim)  # valor por defecto
        else:
            self.limit = limit
            
        # PARÁMETRO E: modification_rate - Factor de modificación de la solución
        self.modification_rate = modification_rate
        
        self.selfun = selfun
        self.extra_params = extra_params

        # asigna propiedades del problema de optimización
        self.evaluate = fun
        self.funcon = funcon
        self.p_f = p_f
        self.lower = lower
        self.upper = upper

        # inicializa el mejor valor actual y su vector solución
        self.best = sys.float_info.max
        self.solution = None

        # crea la colmena con la población de abejas
        self.population = [Bee(lower, upper, fun, funcon) for i in range(self.size)]

        # inicializa la mejor solución como la mejor fuente de néctar encontrada
        self.find_best()

        # calcula las probabilidades de selección
        self.compute_probability()

        # activa o desactiva la visualización del proceso
        self.verbose = verbose

    def stochastic_ranking(self):
        """
        Ordena la población considerando restricciones y función objetivo
        usando jerarquización estocástica.
        """
        # Índices de las abejas
        ranked_indices = list(range(self.size))
        random.shuffle(ranked_indices)

        def compare(i, j):
            bee_i = self.population[i]
            bee_j = self.population[j]

            # Ambos factibles: comparar por función objetivo
            if bee_i.valid and bee_j.valid:
                return bee_i.value - bee_j.value
            # Uno factible, el otro no: el factible es mejor
            elif bee_i.valid:
                return -1
            elif bee_j.valid:
                return 1
            # Ambos infactibles: usar probabilidad p_f
            else:
                if random.random() < self.p_f:
                    # Comparar por función objetivo
                    return bee_i.value - bee_j.value
                else:
                    # Comparar por violación de restricciones
                    return bee_i.violation - bee_j.violation

        # Ordenamiento por burbuja
        for i in range(self.size - 1):
            for j in range(0, self.size - i - 1):
                if compare(ranked_indices[j], ranked_indices[j + 1]) > 0:
                    ranked_indices[j], ranked_indices[j + 1] = ranked_indices[j + 1], ranked_indices[j]

        # Reordenar la población
        self.population = [self.population[i] for i in ranked_indices]

    def find_best(self):
        """
        Aplica jerarquización estocástica para ordenar la población
        y actualiza la mejor solución encontrada hasta ahora
        """
        # Aplica jerarquización estocástica
        self.stochastic_ranking()

        # La mejor abeja queda en la primera posición
        best_bee = self.population[0]

        # Solo actualiza si es una solución válida y mejora
        if best_bee.valid and best_bee.value < self.best:
            self.best = best_bee.value
            self.solution = best_bee.vector[:]  # Crear copia

    def compute_probability(self):
        """
        Calcula la probabilidad relativa de que una solución sea
        elegida por una abeja observadora después de la ceremonia del
        baile del meneo, cuando las abejas empleadas regresan a la colmena.
        """
        # obtiene los valores de aptitud (fitness) de las abejas
        values = [bee.fitness for bee in self.population]
        max_values = max(values)

        # calcula las probabilidades como en la implementación clásica de Karaboga
        if self.selfun is None:
            self.probas = [0.9 * v / max_values + 0.1 for v in values]
        else:
            if self.extra_params is not None:
                self.probas = self.selfun(list(values), **self.extra_params)
            else:
                self.probas = self.selfun(values)

        # devuelve los intervalos acumulados de probabilidad
        return [sum(self.probas[:i+1]) for i in range(self.size)]

    def send_employee(self, index):
        """
        2. FASE DE ABEJAS EMPLEADAS.
        ---------------------------

        En esta segunda fase, se generan nuevas soluciones candidatas
        para cada abeja empleada mediante cruce y mutación.

        Si el nuevo vector mutado resulta mejor que el vector original,
        se reemplaza en la población.
        """
        # copia profundamente el vector solución de la abeja actual
        zombee = copy.deepcopy(self.population[index])

        # selecciona una dimensión aleatoria para mutar
        d = random.randint(0, self.dim-1)

        # selecciona una abeja distinta de la actual
        bee_ix = index
        while bee_ix == index: 
            bee_ix = random.randint(0, self.size-1)

        # realiza la mutación en la dimensión seleccionada
        zombee.vector[d] = self._mutate(d, index, bee_ix)

        # asegura que el vector resultante respete los límites
        zombee.vector = self._check(zombee.vector, dim=d)

        # evalúa la nueva solución (función objetivo y restricciones)
        zombee._evaluate_solution(self.evaluate, self.funcon)

        # aplica selección: se reemplaza si la nueva solución es mejor
        if zombee.fitness > self.population[index].fitness:
            self.population[index] = copy.deepcopy(zombee)
            self.population[index].counter = 0  # reinicia contador si mejora
        else:
            self.population[index].counter += 1  # incrementa contador si no mejora

    def send_onlookers(self):
        """
        3. FASE DE ABEJAS OBSERVADORAS.
        -----------------------

        Se define el mismo número de abejas observadoras que de empleadas.
        Cada observadora intenta mejorar localmente la solución de la abeja
        empleada que decide seguir, después de observar su baile del meneo.
        """
        numb_onlookers = 0
        beta = 0

        while numb_onlookers < self.size:  # una observadora por cada empleada
            # genera un número aleatorio entre 0 y 1
            phi = random.random()

            # incrementa el valor beta de la ruleta de selección
            beta += phi * max(self.probas)
            beta %= max(self.probas)  # mantiene beta dentro del rango

            # selecciona una abeja empleada con base en beta
            index = self.select(beta)

            # la observadora intenta mejorar esa solución
            self.send_employee(index)

            # aumenta el contador de observadoras procesadas
            numb_onlookers += 1

    def select(self, beta):
        """
        4. FASE DEL BAILE DEL MENEO.
        ---------------------

        En esta cuarta fase, las abejas observadoras son reclutadas utilizando
        el método de "selección por ruleta".
        """
        # calcula las probabilidades actualizadas en línea
        probas = self.compute_probability()

        # selecciona una abeja empleada según el valor beta
        for index in range(self.size):
            if beta < probas[index]:
                return index

    def send_scout(self):
        """
        5. FASE DE ABEJA EXPLORADORA.
        -----------------------

        Identifica a las abejas cuyo contador de intentos supera el límite predefinido,
        las reemplaza y crea una nueva abeja aleatoria que explore otra región del
        espacio de búsqueda.
        """
        # obtiene el número de intentos de cada abeja
        trials = [self.population[i].counter for i in range(self.size)]

        # identifica la abeja con el mayor número de intentos sin mejora
        index = trials.index(max(trials))

        # verifica si esa abeja supera el límite permitido de intentos (usando el parámetro 'limit')
        if trials[index] > self.limit:
            # crea una nueva abeja aleatoria (exploradora)
            self.population[index] = Bee(self.lower, self.upper, self.evaluate, self.funcon)

            # envía a la nueva exploradora a intentar mejorar su vector
            self.send_employee(index)

    def _mutate(self, dim, current_bee, other_bee):
        """
        Realiza la mutación de una dimensión del vector solución (para valores continuos).
        Utiliza el parámetro modification_rate para controlar la intensidad de la mutación.

        Parámetros:
        ----------
            :param int dim         : dimensión del vector a mutar
            :param int current_bee : índice de la abeja actual
            :param int other_bee   : índice de otra abeja para aplicar cruce/mutación
        """
        # Factor phi aleatorio entre -modification_rate y +modification_rate
        phi = (random.random() - 0.5) * 2 * self.modification_rate
        
        return (self.population[current_bee].vector[dim] + 
                phi * (self.population[current_bee].vector[dim] - self.population[other_bee].vector[dim]))

    def _check(self, vector, dim=None):
        """
        Verifica que un vector solución se mantenga dentro de los
        límites inferiores y superiores predefinidos del problema.
        """
        if dim is None:
            range_ = range(self.dim)
        else:
            range_ = [dim]

        for i in range_:
            # verifica el límite inferior
            if vector[i] < self.lower[i]:
                vector[i] = self.lower[i]
            # verifica el límite superior
            elif vector[i] > self.upper[i]:
                vector[i] = self.upper[i]

        return vector

    def _verbose(self, itr, cost):
        """ Muestra información sobre el proceso de optimización. """
        msg = "# Iteración = {} | Mejor valor de la evaluación = {} | Valor medio de evaluación = {} "
        print(msg.format(int(itr), cost["best"][itr], cost["mean"][itr]))

# ---- FIN