
Training

docker run -it --rm -v "{dataset}":/app/data:ro -v "{checkpoint-output}":/app/checkpoints buildingfootprints/segmentation:0.1
    python train.py
    --num_epochs 100 --crop_height 500 --crop_width 500 --h_flip True --v_flip True --brightness 0.2 --rotation 180 --model MobileUNet --batch_size 2 --frontend ResNet50
   
Testing

docker run -it --rm -v "{dataset}":/app/data:ro -v "{checkpoint-output}":/app/checkpoints:ro -v "{test-result}":/app/Test buildingfootprints/segmentation:0.1
   python test.py
    --crop_height 500 --crop_width 500 --model MobileUNet
    
Predicting
docker run -it --rm -v "{dataset}":/app/data:ro 
                    -v "{checkpoint}":/app/checkpoints:ro 
                    -v "{image-dir}:/app/Predict"
                    buildingfootprints/segmentation:0.1
                    python predict.py
                    --image Predict/{filename}
                    --crop_height 500
                    --crop_width 500
                    --model MobileUNet