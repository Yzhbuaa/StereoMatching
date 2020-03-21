import time
import inspect


def timer():
    start = time.time()

    def end(function_name = "Unnamed function"):
        print(function_name + " took " + str(time.time() - start) + " seconds. ")
        return

    return end
