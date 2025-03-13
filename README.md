# DoctorAssign

AI chatbot which assigns a doctor based on the user's needs and location.


Backend Setup instructions.

1. Run this command to setup IRIS Database. Make sure you have docker desktop installed.
docker run -d --name iris-doctorassign -p 1972:1972 -p 52773:52773 \
    -e IRIS_PASSWORD=demo -e IRIS_USERNAME=demo \
    intersystemsdc/iris-community:latest

2. Run the Docker Desktop and turn on the server., there is a play button when you open the docker dashboard. there should be a new database called iris-doctorassign. Whenever u want to run the backend docker must be running, or the database won't update.

3. Open terminal or cmd and go to the simpleflask directory. Use "Python import_backend_schema.py" to import the csv files to the database. (Docker must be running)

4. Use "Python flaskBackend.py" to turn on the backend. Now it is ready to receive requests. 