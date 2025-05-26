# Blur Kernel Size Estimation in Blind Image Deblurring

This repository contains MATLAB implementations related to the research paper submitted to the *بینایی ماشین و پردازش تصویر* journal:

## 📄 Title  
روشی کارآمد برای تخمین ابعاد هسته‌ی تاری در رفع تاری کور تصویر  

## ✍️ Authors  
Mitra Abdollahi and Alireza Ahmadyfard  

## 📧 Emails  
mitra.abdollahi@shahroodut.ac.ir  
ahmadyfard@shahroodut.ac.ir  

---

## 📖 Overview

This project addresses the problem of estimating the blur kernel size in blind image deblurring. The repository provides:

- **Proposed Method**: Our novel approach for efficient and accurate blur kernel size estimation.
- **Ref 18 Method**: Our independent reimplementation of the method described in [18], as no original code was provided by the authors.

---

## 📁 Repository Structure

```
blur-kernel-size-estimation-in-BID/
├── Proposed Method/           # Implementation of the proposed approach
│   └── RUN_proposed.m        # Main script to run the proposed method
├── Ref 18 Method/            # Reimplementation of the method from Reference [18]
│   └── RUN_Ref18.m           # Main script to run the Ref 18 method
├── dataset/                  # Dataset folder (not included in repo, see below)
│   ├── gaussian_symmetric/
│   ├── gaussian_without_tilt/
│   ├── gaussian_tilt/
│   └── non_gaussian/
└── README.md                 # This documentation file
```

The `dataset` folder includes subfolders for different types of blur kernels and contains images used for training and testing the methods.

---

## 🛠️ Requirements and Environment

- MATLAB **R2021a**  
- Tested on a system with:  
  - **Intel Core i7-6700HQ** CPU @ 2.6 GHz  
  - **12 GB RAM**

---

## 🚀 How to Run

1. Clone or download this repository.
2. Download and extract the dataset (see next section), then place the `dataset` folder inside the root directory of this repository.
3. Open MATLAB and navigate to the appropriate folder.
4. To run the **Proposed Method**, execute the following in MATLAB:
   ```matlab
   cd 'path_to/blur-kernel-size-estimation-in-BID/Proposed Method'
   run('RUN_proposed.m')
   ```
5. To run the **Ref 18 Method**, execute:
   ```matlab
   cd 'path_to/blur-kernel-size-estimation-in-BID/Ref 18 Method'
   run('RUN_Ref18.m')
   ```

---

## 📂 Dataset

Due to GitHub file size limitations (~2.4 GB), the dataset is **not included** in this repository.  
You can download it from the following Google Drive link:

🔗 [Download Dataset](https://drive.google.com/file/d/17XcknkSf-L3OHNRi-TjLAOnN_lHbn8RH/view?usp=sharing)

### 📌 Instructions:
- Download and extract the dataset.
- Place the extracted `dataset` folder inside the root of this repository.
- The `dataset` folder includes subfolders for different types of blur kernels and corresponding images used for training and testing the algorithms.

---

## 📚 Reference

[18] S. Liu, H. Wang, J. Wang, and C. Pan,  
"Blur-Kernel Bound Estimation from Pyramid Statistics,"  
*IEEE Transactions on Circuits and Systems for Video Technology*, vol. 26, no. 5, pp. 1012–1016, May 2016.  
🔗 [https://doi.org/10.1109/TCSVT.2015.2418585](https://doi.org/10.1109/TCSVT.2015.2418585)

---

## 📢 Important Notes for Reviewers and Users

- The **Ref 18 Method** is **our own independent reimplementation** of the method described in Reference [18].
- No original code was obtained from the authors of [18]; we re-implemented their method ourselves.
- If you use or build upon this code, please **cite our paper**.

---

## ⚖️ License and Copyright

This software is provided **for academic and research use only**.  
Redistribution, commercial use, or modification **without proper citation is prohibited**.  
Users **must cite the original paper** when using this code in their research.  
All rights are reserved by the authors.

---

## 📬 Contact

For any questions or collaboration opportunities, please contact:  
📧 **Mitra Abdollahi** – mitra.abdollahi@shahroodut.ac.ir  
📧 **Alireza Ahmadyfard** – ahmadyfard@shahroodut.ac.ir
