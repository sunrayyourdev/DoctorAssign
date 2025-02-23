# Developer Instructions

## Package Management

### Installing Packages

Run the following command to install Python dependencies:

```bash
pip install -r requirements.txt
```

### Adding New Packages

To install a new package and update your dependencies, follow these steps:

1. Install the package:

```bash
pip install <package-name>
```

2. Freeze the updated dependencies:

```bash
pip freeze > requirements.txt
```

## Running the App

Start the Python server with:

```bash
python app.py
```

## Using test.http

Use the provided test.http file to quickly test API endpoints using a REST client (e.g., the REST Client extension in VSCode).
