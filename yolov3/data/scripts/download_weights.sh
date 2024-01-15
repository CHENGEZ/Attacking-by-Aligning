python - <<EOF
from utils.downloads import attempt_download

models = ['yolov3', 'yolov3-spp', 'yolov3-tiny']
for x in models:
    attempt_download(f'{x}.pt')

EOF
