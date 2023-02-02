# OKCE
## 📔 Algorithm
* Flow
![algorithm](./Images/OKCE.png)
* Performance
![algorithm](./Images/OKCE_performance.png)
* Relations
![algorithm](./Images/OKCE_rel.png)

## ⌨️ Usage
1. Dependencies  
```python
python==3.8
scikit-learn==1.1.1
torch==1.12.1
```
## 🗂 Major DIR
1. [data](./data)
* learner logs
2. [PGM](./PGM)
* Code for KDD dataset
* merge_main_nonFS.py  
    * No Feature Selection with DKT  
    ```python
    python merge_main_nonFS.py --data_path=../data/kc_dedup_smath11_reshape.csv --n_epochs=150
    ```
* merge_main.py  
    * Feature Selection with DKT  
    ```python
    python merge_main.py --data_path=../data/kc_dedup_smath11_reshape.csv --n_epochs=150
    ```
3. [SSM_PGM](./SSM_PGM)
* Code for SSM dataset
4. [json](./json)
* meta files