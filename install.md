### Prerequisites:
- this clone this repository to your machine
- Install docker for you computer : https://docs.docker.com/engine/install/


Once the code is cloned and docker is installed:

- Build a docker image with this command: `` docker build -t trudenty_test . ``
- Verify the image is built: `` docker images
 `` . You should see trudenty_test image under the REPOSITORY column
- Now that you have built the image, you can run the container by executing: ``docker run -p 8501:8501 trudenty_test``. If all went well, you should see an output similar to the following:
 ```
 docker run -p 8501:8501 trudenty_test

  You can now view your Streamlit app in your browser.

  URL: http://0.0.0.0:8501
  ```
