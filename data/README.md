# Persona data

### 1. .tar -> .zip
- find "폴더경로" -name "파일명.zip.part*" -print0 | sort -zt'.' -k2V | xargs -0 cat > "파일명.zip"
ex) find /data1/home/dylan16/kt_track1/demo_agent -name download_shopping.tar -print0 | sort -zt'.' -k2V | xargs -0 cat > data_shopping.zip

### 2. unzip
