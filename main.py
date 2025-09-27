import numpy as np
import pandas as pd


def main():
    points = [0.0]
    for i in range(1, 500):
        points.append(points[i] + np.random.normal(0.0, 1.0))
    
    

if __name__ == "__main__":
    main()
