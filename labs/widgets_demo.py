import streamlit as st

st.title("Interactive Widgets Demo")

# Text input
name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}!")


# Number input
age = st.number_input("Enter your age:", min_value=0, max_value=100, value=25)
st.write(f"You are {age} years old")

# Slider
score = st.slider("Rate your Python skills (1-10):", 1, 10, 5)
st.write(f"Your rating: {score}/10")

# Select box
favorite_color = st.selectbox("Choose your favorite color:", 
                             ["Red", "Blue", "Green", "Yellow"])
st.write(f"Your favorite color is {favorite_color}")

# Checkbox
if st.checkbox("Show secret message"):
    st.write("🎉 You found the secret!")