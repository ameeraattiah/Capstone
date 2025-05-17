// firebase.js
import { initializeApp } from "https://www.gstatic.com/firebasejs/9.6.4/firebase-app.js";
import {
  getAuth,
  GoogleAuthProvider,
  GithubAuthProvider, // ✅ ADD THIS
  signInWithPopup,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  onAuthStateChanged
} from "https://www.gstatic.com/firebasejs/9.6.4/firebase-auth.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/9.6.4/firebase-firestore.js";

// Firebase config
const firebaseConfig = {
  apiKey: "AIzaSyDtqyEQIhESMTbbW0nyZMmCjrLy451s6sI",
  authDomain: "nlpulse-c22af.firebaseapp.com",
  projectId: "nlpulse-c22af",
  storageBucket: "nlpulse-c22af.appspot.com",
  messagingSenderId: "562893359272",
  appId: "1:562893359272:web:2d3d85b8905baa3c3eda9d",
  measurementId: "G-T7862EJRWR"
};

const app = initializeApp(firebaseConfig);

export const auth = getAuth(app);
export const db = getFirestore(app);
export const provider = new GoogleAuthProvider();
export const githubProvider = new GithubAuthProvider(); // ✅ THIS LINE IS REQUIRED
