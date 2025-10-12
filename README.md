# Machine Learning for Higher Order Optical Spatial Modes in Turbulence using Python and Tensorflow
This project uses machine learning to aid in turbulence affected optical communication and networks. This code includes simulation in python for generating higher order Ince, Hermite, and Laguerre Gaussian modes. 

1) Generation of higher order Ince, Hermite, and Laguerre Gaussian modes.
2) Add turbulence
3) Model propagation 
4) Generate data-sets for machine learning projects
5) Use generative models for denoising spatial modes
6) Use classifying convolutional neural networks for mode classification

If you use this code, we ask that you cite:

Manon P. Bart, Sita Dawanse, Nicholas J. Savino, Viet Tran, Tianhong Wang, Sanjaya Lohani, Farris Nefissi, Pascal Bassène, Moussa N’Gom, and Ryan T. Glasser, "Classification of single photons in higher-order spatial modes via convolutional neural networks," Opt. Lett. 50, 2820-2823 (2025)

## Table of Contents
- [Introduction](#introduction)
- [Usage](#usage)
- [License](#license)


## Introduction
This project simulates and analyzes higher-order spatial modes of light for use in turbulence-affected optical communication. The main focus is on Hermite-Gaussian (HG), Laguerre-Gaussian (LG), and Ince-Gaussian (IG) beams, their propagation, and their interaction with turbulence.

- **Hermite-Gaussian (HG) modes**  
  Defined in Cartesian coordinates with rectangular symmetry.

  <img width="555" height="47" alt="Screenshot 2025-10-01 at 6 07 15 PM" src="https://github.com/user-attachments/assets/f2d5dfe8-0561-44f4-8ac9-d00e1c80f5a2" />

 where $H_n$ are Hermite polynomials, $\omega(z)$ is beam width, and $\xi(z)$ is the Gouy phase.

- **Laguerre-Gaussian (LG) modes**  
  Defined in cylindrical coordinates with circular symmetry and orbital angular momentum (OAM).

    <img width="566" height="55" alt="Screenshot 2025-10-01 at 6 07 38 PM" src="https://github.com/user-attachments/assets/ff785ead-feb8-42bb-bcfd-ddc0bb3ec6ba" />

  where $L_n^{|\ell|}$ are generalized Laguerre polynomials, $\ell$ is the azimuthal index (OAM).

- **Ince-Gaussian (IG) modes**  
  Defined in elliptical coordinates with an ellipticity parameter $\epsilon$. They interpolate smoothly between HG and LG families.

   <img width="463" height="57" alt="Screenshot 2025-10-01 at 6 07 56 PM" src="https://github.com/user-attachments/assets/43123eab-cae9-493a-ab9a-b288e59df8f5" />

  where $\mathcal{C}_{p,m}^{e/o}$ are even/odd Ince polynomials and $\epsilon$ controls ellipticity.

---

### Turbulence Modeling
Atmospheric turbulence is simulated using the Kolmogorov/Von Kármán model. 

<img width="251" height="61" alt="Screenshot 2025-10-01 at 6 08 19 PM" src="https://github.com/user-attachments/assets/969eb809-86c2-4722-88bc-ce7cd6be5ecc" />


The turbulence strength is quantified by the structure constant $C_n^2$:  

- **Weak turbulence**: $C_n^2 \approx 10^{-17}\,\text{m}^{-2/3}$  
- **Moderate turbulence**: $C_n^2 \approx 10^{-15}\,\text{m}^{-2/3}$  
- **Strong turbulence**: $C_n^2 \approx 10^{-13}\,\text{m}^{-2/3}$ or higher  

A smaller inner scale $l_0$ means stronger distortions, while the outer scale $L_0$ sets the largest turbulence features.

---

### Propagation
Beam propagation is modeled with the Angular Spectrum Method (AS). This method decomposes the field into plane waves, propagates each independently, and reconstructs the beam after a chosen distance. It is efficient and well-suited for simulating free-space optical communication.

---

In summary, this repository lets you:
1. Generate HG, LG, and IG modes of arbitrary order.  
2. Propagate them using the Angular Spectrum Method.  
3. Add turbulence to model realistic free-space communication conditions.  
4. Build datasets for training denoising and classifying machine learning models for mode recognition and classification.

## Usage
Generation of modes:

Propagation:

Machine Learning:
The data can be generated as follows-

Run the model using-

## Liscence


