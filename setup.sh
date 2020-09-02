# copied from taxifaremodel not sure what to put for the email address

mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"chris.westerman8@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
