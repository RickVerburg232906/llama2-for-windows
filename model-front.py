import streamlit as st
import zmq
import yaml
import time

with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

context = zmq.Context()

# Connecting to socket (with retry) 
connected = False

while not connected:
    try:
        socket = context.socket(zmq.REQ)
        socket.connect(config['socket']['REQ'])
        connected = True
    except Exception as e:
        time.sleep(5)

st.title('[ Talk with Meta-Llama2 ]')
user_input = st.text_input("Input:")

if st.button("Send"):
    socket.send_string(user_input)
    st.write(socket.recv_string())
