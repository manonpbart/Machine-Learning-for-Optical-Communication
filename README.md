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
- **Strong turbulence**: $C_n^2 \approx 10^{-13}\,\text{m}^{-2/3}$

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

### Generation of modes:

Information for generating modes can be found with further detail in beam_project/examples/GeneratingBeams.ipynb. Once the beam_simulation package is downloaded, many different optical set ups can be configured using:

          import beam_simulation as bs
          bs.Config(wavelength=795e-9, size_x=400, size_y=400, pixel_size=8e-6,w0=0.35e-3)

Following this, different beams can be generated using:

        E_lg = bs.lg(p=2, l=1)

<img width="364" height="372" alt="d7595316-86b5-4dc9-a830-4c96a243b444" src="https://github.com/user-attachments/assets/92f2203c-b2b4-47e7-aa72-ddc6fa5b34a2" />

        E_hg = bs.hg(n=2, m=3)
        
<img width="364" height="372" alt="cb9825b1-19dc-44ee-856a-5f65568116c9" src="https://github.com/user-attachments/assets/a845aee0-ad8d-4b8d-96b4-af74ce843e6c" />

        E_ig = bs.ig(p=4, m=2, beam="e", elliptic_param=2)
        
<img width="364" height="372" alt="566763b6-9ac1-43a1-8ed6-7b63d69ac824" src="https://github.com/user-attachments/assets/089669f8-f559-4d94-88bd-bb9e805ab813" />

where the elliptic parameter as well as even or odd modes can also be changed. 

Propagation of modes can be done using:

        E_prop = bs.propagation(E_lg, z=20e-2)

Turbulence can be added for varying Cn2 values:

        phi_turb = bs.turbulence(Cn2=38e-12, l_max=25, l_min=1e-3)
        E_turb = E_lg * np.exp(1j * phi_turb)
        E_turb_prop = bs.propagation(E_turb, z=20e-2)

<img width="1176" height="407" alt="08c35939-f62b-4a54-871d-0a546ab90dea" src="https://github.com/user-attachments/assets/acc518cc-9909-4196-b170-b766abb71ff2" />


### Machine Learning:

For machine learning tasks, large data sets must be created. Examples of this process is shown in beam_project/examples/GenerativeModel.ipynb

Using the beam simulation package, data sets can be created using 

        import beam_simulation as bs
        from beam_simulation.create_oam_list import GenerateOAM
        gen = GenerateOAM(
            mode_types=["LG"],
            orders={
                "LG": {"p": [0,1,1], "l": [1,2,1]}, #pairs of order numbers
                "HG": {"n": [0, 1], "m": [1, 2]}
            },
            Cn2_list=[71.8e-13, 30.2e-12, 12e-12], #different turbulence levels
            n_samples=400 #number of images per mode and order
        )

which can generate different pairs of LG, HG, and IG modes as well as change the number of images and the different turbulence strengths. The mode and order numbers can be visualized using:

        gen.visualize("LG", "p1_l2", Cn2=71.8e-13)

<img width="795" height="280" alt="d740a282-dfc4-48dc-ad6f-7441108470b0" src="https://github.com/user-attachments/assets/435cdf41-22da-4477-bfb8-93e268594caa" />

In the GenerativeModel code, there is a sample denoising convolutional autoencoder which reduces the effects of turbulence on input images. This can be called using:

        autoenc = ConvAutoencoder(
            input_shape=images.shape[1:],
            latent_dim=64, #size of latent vector
            conv_filters=[32, 16, 8], #determines filter sizw and number of filters
            use_batchnorm=True, #batch normalization helps with normalizing during training
            use_pool=True,
            pool_size = 2,
            pool_where = [0], #pool where determines where to add a pooling layer
            decoder_activation='linear'
        )
        
        #Splitting everything
        x_train, x_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42, shuffle=True)
        
        #Training
        history = autoenc.train(x_train, y_train, x_test, y_test, epochs=15, batch_size=32)

The images can then be reconstructed using:

        xtest_decoded = autoenc.reconstruct(x_test)

Outputs of the denoising CAE and the input images are shown here:

<img width="1414" height="368" alt="c1a020bd-7362-44c8-ac14-f0c1189f5432" src="https://github.com/user-attachments/assets/ad4cb06f-8bdf-48f7-a999-8b67d2991078" />


## Liscence


