# SelfSVT

Code and Environments for  A Self-Supervised Solution for the Switch-Toggling Visual Task  

## Environments

We test the performance of SelfSVT on two environments: a simulated environment (

[EnS]: https://github.com/StanfordVL/causal_induction

) proposed by Nair et al. and a real-world environment found by ourselves (EnR). EnR is a typical conference hall that can accommodate at least 200 listeners.



#### The Original Data

For each environment, we collect snapshots under all the possible switch states. 

If you want to get these original data, please send an email to fulva.hyh@gmail.com.



#### The Processed Data

To simplify our experiments, all the data is processed and stored as NPY files. 

For example:  EnR_all_images_N5_Size32.npy:

- EnR is the name of the environment
- N is the number of switches and N5  means that the number of switches is 5 
- Size is the size of the input image and Size32 means that the size of our input image is (32, 32, 3) 

We could following code to read our NPY files:

```python
data_dict = np.load(os.path.join(dirpath, args.scene+"_all_images_N"+str(args.num)+"_Size"+str(args.w_size)+".npy"), allow_pickle=True).item()
```

data_dict is a dict. The key is the switch state and the value is the image data. 

If you want to test with your environment，you could transfer your data to this format.



## Model Training & Evaluation

Train model with 30% snapshots under environment EnS with 7 switches 

```python
python main.py --w-size 32 --num 7 --seen 30 --scene EnS
```

It will output training and test results every 1000 epochs. 

