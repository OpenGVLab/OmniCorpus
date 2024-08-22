from llava.train.train_interleaved import train

if __name__ == "__main__":
    # # OSError: IOError: broken data stream when reading image file
    # from PIL import Image
    # from PIL import ImageFile
    # ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    train(attn_implementation="flash_attention_2")
