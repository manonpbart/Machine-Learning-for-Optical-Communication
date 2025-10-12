import numpy as np
import beam_simulation as bs
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

class GenerateOAM:
    """
    Generate OAM beam datasets (LG, HG, IG) with turbulence.

    Args:
        mode_types (list[str]): list of mode families to include ['LG', 'HG', 'IG'].
        orders (dict): mapping per mode, e.g. {'LG': {'p':[0,1], 'l':[1,2]}, 'HG': {'n':[0,1], 'm':[1,2]}}
        Cn2_list (list[float]): turbulence levels to simulate.
        n_samples (int): number of realizations per condition.
        size (int): grid size (pixels).
    """

    def __init__(self, mode_types, orders, Cn2_list, n_samples=10):
        self.mode_types = mode_types
        self.orders = orders
        self.Cn2_list = Cn2_list
        self.n_samples = n_samples
        self.data = self._build_dataset()

    # -------------------------------------------------
    # Main dataset generation
    # -------------------------------------------------
    def _build_dataset(self):
        data = {}

        for mode in self.mode_types:
            data[mode] = {}

            #Determine order parameter sets
            order_params = self.orders[mode]
            order_list = self._get_order_pairs(order_params)

            for order_name, params in order_list:
                data[mode][order_name] = {}
                
                #Generate beam depending on mode
                E0 = self._generate_beam(mode, **params)

                for Cn2_val in self.Cn2_list:
                    data[mode][order_name][Cn2_val] = []

                    for _ in range(self.n_samples):
                        turb = bs.turbulence(Cn2=Cn2_val, l_max=25, l_min=1e-3)

                        #Apply turbulence and propagate
                        E_turb = E0 * np.exp(1j * turb)
                        E_prop = bs.propagation(E_turb, z=0.4)

                        #Normalize and store intensity
                        img = np.abs(E_prop) ** 2
                        img /= np.max(img)

                        data[mode][order_name][Cn2_val].append(img)

        return data

    # -------------------------------------------------
    # Mode-specific beam creation
    # -------------------------------------------------
    def _generate_beam(self, mode, **params):
        if mode.upper() == "LG":
            return bs.lg(p=params["p"], l=params["l"])
        elif mode.upper() == "HG":
            return bs.hg(n=params["n"], m=params["m"])
        elif mode.upper() == "IG":
            return bs.ig(p=params["p"], m=params["m"], beam="e")
        else:
            raise ValueError(f"Unknown mode {mode}")

    # -------------------------------------------------
    # Convert dict of order lists into (label, params) pairs
    # -------------------------------------------------
    def _get_order_pairs(self, order_dict):
        keys = list(order_dict.keys())

        # for LG/IG — expect p and l or p and m
        if len(keys) == 2:
            key1, key2 = keys
            combos = []
            for val1 in order_dict[key1]:
                for val2 in order_dict[key2]:
                    label = f"{key1}{val1}_{key2}{val2}"
                    combos.append((label, {key1: val1, key2: val2}))
            return combos

        # for HG — expect n and m
        elif len(keys) == 2:
            key1, key2 = keys
            combos = [(f"{key1}{n}_{key2}{m}", {key1: n, key2: m})
                      for n in order_dict[key1] for m in order_dict[key2]]
            return combos

        else:
            raise ValueError("Order dict must include two parameters (e.g., p,l or n,m).")

    # -------------------------------------------------
    # Optional visualization 
    # -------------------------------------------------
    def visualize(self, mode="LG", order="p0_l1", Cn2=None):
        Cn2 = Cn2 or self.Cn2_list[-1]
        imgs = self.data[mode][order][Cn2]
        plt.figure(figsize=(10, 4))
        for i, img in enumerate(imgs[:5]):
            plt.subplot(1, 5, i + 1)
            plt.imshow(img, cmap="plasma")
            plt.axis("off")
        plt.suptitle(f"{mode} {order} @ Cn2={Cn2:.1e}")
        plt.show()