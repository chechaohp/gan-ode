cd $1
wget https://storage.googleapis.com/thumos14_files/UCF101_videos.zip
unzip UCF101_videos.zip
rm -rf UCF101_videos.zip
python3 utils/classify_video.py UCF101 UCF101/videos_classified
wget https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
unzip UCF101TrainTestSplits-RecognitionTask.zip
mv ucfTrainTestlist UCF101/annotations
rm -rf UCF101TrainTestSplits-RecognitionTask.zip