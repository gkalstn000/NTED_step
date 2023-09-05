# train.py 프로세스 종료를 위해 PID 가져오기
PROCESS_NAME=$1

PID=$(ps -ef | grep "$PROCESS_NAME" | grep -v grep | awk '{print $2}')
echo $PID
