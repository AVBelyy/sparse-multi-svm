import numpy as np
import fibonacci_heap_mod as fhm
from scipy.sparse.linalg import norm

class GradientUpdateTool(object):
    """
    Superclass for gradient routine maintainers in coordinate descent methods
    """

    def get_coordinate(self):
        raise Exception("Not implemented yet")

    def get_value(self):
        raise Exception("Not implemented yet")

    def update(self, delta):
        raise Exception("Not implemented yet")

    def get(self):
        raise Exception("Not implemented yet")

    def get_norm(self):
        raise Exception("Not implemented yet")


class BasicGradientUpdater(GradientUpdateTool):

    def __init__(self, g):
        self.__g = g

    def get_coordinate(self):
        return self.__g.argmin()

    def get_value(self):
        return self.__g.min()

    def update(self, delta):
        self.__g += delta

    def get(self):
        return self.__g

    def get_norm(self):
        return norm(self.__g)


class HeapGradientUpdater(GradientUpdateTool):

    def __init__(self, g):
        self.__g = g
        self.__g_norm = 0
        self.__heap = fhm.Fibonacci_heap()
        self.__g_elements = []

        if max(g.shape) <= 1e8:    # dense vectors work significantly better if not blow memory
            for i, val in enumerate(np.squeeze(g.toarray())):
                self.__g_elements.append(self.__heap.enqueue(i, val))
                self.__g_norm += val**2

        else:
            raise Exception("This method hasn't been tested yet on dimensions n > 10^8, sorry :(")

    def update(self, delta):
        g_elems = self.__g_elements
        heap = self.__heap

        for k in delta.nonzero()[1]:
            old_priority = g_elems[k].get_priority()
            new_priority = old_priority + delta[0, k]
            if old_priority > new_priority:
                heap.decrease_key(entry=g_elems[k], new_priority=new_priority)

            else:
                value = g_elems[k].get_value()
                heap.decrease_key(entry=g_elems[k], new_priority=heap.min().get_priority() - 1)
                heap.dequeue_min()
                g_elems[k] = heap.enqueue(value=value, priority=new_priority)
            self.__g_norm = self.__g_norm - old_priority ** 2 + new_priority ** 2

        self.__g += delta

    def get(self):
        return self.__g

    def get_norm(self):
        return self.__g_norm

    def get_coordinate(self):
        return self.__heap.min().get_value()

    def get_value(self):
        return self.__heap.min().get_priority()