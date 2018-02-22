import os, shutil

def mkdir_(path):
    if os.path.isdir(path) == False:
        os.mkdir(path)

def prepare_data() :
    original_dataset_dir = "/nfs/site/disks/infra_work_06/pmangala/ML/datasets/train"
    target_base_dir = "/nfs/site/disks/infra_work_06/pmangala/ML/datasets/cats_n_dogs/"
    
    if os.path.isdir(target_base_dir) == False :
        mkdir_(target_base_dir)

    train_dir = os.path.join(target_base_dir, 'train')
    mkdir_(train_dir)
    val_dir= os.path.join(target_base_dir, 'val')
    mkdir_(val_dir)
    test_dir = os.path.join(target_base_dir, 'test')
    mkdir_(test_dir)

    train_dogs_dir = os.path.join(train_dir, 'dogs')
    mkdir_(train_dogs_dir)
    train_cats_dir = os.path.join(train_dir, 'cats')
    mkdir_(train_cats_dir)

    val_dogs_dir = os.path.join(val_dir, 'dogs')
    mkdir_(val_dogs_dir)
    val_cats_dir = os.path.join(val_dir, 'cats')
    mkdir_(val_cats_dir)

    test_dogs_dir = os.path.join(test_dir, 'dogs')
    mkdir_(test_dogs_dir)
    test_cats_dir = os.path.join(test_dir, 'cats')
    mkdir_(test_cats_dir)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copy(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(val_cats_dir, fname)
        shutil.copy(src, dst)
 
    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copy(src, dst)


    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copy(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(val_dogs_dir, fname)
        shutil.copy(src, dst)
 
    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copy(src, dst)
        
#prepare_data()
