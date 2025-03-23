# Portfolio site

## Experience level
- My first web development experience outside of a structured course.
- I am learning as I continue to develop.
- This is my first time using html, flask, and css.

## Installation and Executing
1. `git clone` or `fork` this repository
2. Go into the folder that contains the contents of this project
3. `python -m venv venv`
4. `source venv/bin/activate`  -- Linux only
5. `pip install -r requirements.txt`
6. `python main.py`


## Project details
- The project includes a `.service` file with which this web application could be served, publicly.
- The `server_setup.sh` is a script that is designed to be executed line-by-line, outside of the script.  
  - The `server_setup.sh` contains instructions that will turn a linux computer, specifically `Ubuntu`, into a web server up to setting up nginx and getting a `ssl` cert.
  - The `ssl` certificate requires a domain name which to register the certificate.
  - Items to do after instructions in the `server_setup.sh` are executed are to connect the domain name provider's `DNS name servers` to the `public IP address` of the linux server that is serving the flask application.

