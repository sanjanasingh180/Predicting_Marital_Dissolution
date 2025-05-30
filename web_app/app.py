import streamlit as st
import requests

st.title("Divorce Prediction System")

st.subheader("Enter Responses to the 54 Questions")

# Explanation of response scale
st.markdown("**Response Scale:**")
st.markdown("0 = Never, 1 = Seldom, 2 = Averagely, 3 = Frequently, 4 = Always")

# Questions with feature numbers
questions = [
    "If one of us apologizes when our discussion deteriorates, the discussion ends.",
    "I know we can ignore our differences, even if things get hard sometimes.",
    "When we need it, we can take our discussions with my spouse from the beginning and correct it.",
    "When I discuss with my spouse, to contact him will eventually work.",
    "The time I spent with my wife is special for us.",
    "We don't have time at home as partners.",
    "We are like two strangers who share the same environment at home rather than family.",
    "I enjoy our holidays with my wife.",
    "I enjoy traveling with my wife.",
    "Most of our goals are common to my spouse.",
    "I think that one day in the future, when I look back, I see that my spouse and I have been in harmony with each other.",
    "My spouse and I have similar values in terms of personal freedom.",
    "My spouse and I have similar sense of entertainment.",
    "Most of our goals for people (children, friends, etc.) are the same.",
    "Our dreams with my spouse are similar and harmonious.",
    "We're compatible with my spouse about what love should be.",
    "We share the same views about being happy in our life with my spouse.",
    "My spouse and I have similar ideas about how marriage should be.",
    "My spouse and I have similar ideas about how roles should be in marriage.",
    "My spouse and I have similar values in trust.",
    "I know exactly what my wife likes.",
    "I know how my spouse wants to be taken care of when she/he is sick.",
    "I know my spouse's favorite food.",
    "I can tell you what kind of stress my spouse is facing in her/his life.",
    "I have knowledge of my spouse's inner world.",
    "I know my spouse's basic anxieties.",
    "I know what my spouse's current sources of stress are.",
    "I know my spouse's hopes and wishes.",
    "I know my spouse very well.",
    "I know my spouse's friends and their social relationships.",
    "I feel aggressive when I argue with my spouse.",
    "When discussing with my spouse, I usually use expressions such as ‘you always’ or ‘you never’ .",
    "I can use negative statements about my spouse's personality during our discussions.",
    "I can use offensive expressions during our discussions.",
    "I can insult my spouse during our discussions.",
    "I can be humiliating when we discuss.",
    "My discussion with my spouse is not calm.",
    "I hate my spouse's way of opening a subject.",
    "Our discussions often occur suddenly.",
    "We're just starting a discussion before I know what's going on.",
    "When I talk to my spouse about something, my calm suddenly breaks.",
    "When I argue with my spouse, I only go out and I don't say a word.",
    "I mostly stay silent to calm the environment a little bit.",
    "Sometimes I think it's good for me to leave home for a while.",
    "I'd rather stay silent than discuss with my spouse.",
    "Even if I'm right in the discussion, I stay silent to avoid hurting my spouse.",
    "When I discuss with my spouse, I stay silent because I am afraid of not being able to control my anger.",
    "I feel right in our discussions.",
    "I have nothing to do with what I've been accused of.",
    "I'm not actually the one who's guilty about what I'm accused of.",
    "I'm not the one who's wrong about problems at home.",
    "I wouldn't hesitate to tell my spouse about her/his inadequacy.",
    "When I discuss, I remind my spouse of her/his inadequacy.",
    "I'm not afraid to tell my spouse about her/his incompetence."
]

input_features = []
for i, question in enumerate(questions, start=1):
    value = st.selectbox(f"Feature {i}: {question}", options=[0, 1, 2, 3, 4], index=2)
    input_features.append(value)

# Convert input to JSON
data = {"features": input_features}

if st.button("Predict"):
    response = requests.post("http://127.0.0.1:5000/predict", json=data)

    if response.status_code == 200:
        result = response.json()
        st.write("### Prediction:", "Divorced" if result["prediction"] == 1 else "Not Divorced")
        st.write("### Probability:", round(result["probability"] * 100, 2), "%")
    else:
        st.error(f"Error: {response.json().get('error', 'Unknown error')}")
