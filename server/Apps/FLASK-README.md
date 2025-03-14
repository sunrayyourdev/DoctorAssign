## simpleFlask

1. Change directory to the project folder:

    ```bash
    cd Apps/simpleFlask
    ```

2. Install the dependencies:

    ```bash
    pip install --no-cache-dir -r requirements.txt 
    ```
    
    Also make sure that you have followed step 5 to install the db-api driver from [`README.md`](../README.md) 
    
3. Freeze new dependencies:
    ```bash
    pip freeze > requirements.txt
    ```

4. Start the Flask application:
   ```bash
   python flaskBackend.py
   ```

5. Open your browser and go to:

    [http://localhost:5010](http://localhost:5010)

![alt text](image.png)
