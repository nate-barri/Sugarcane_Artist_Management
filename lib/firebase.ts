// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getAnalytics } from "firebase/analytics";

// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyBOHOw4jZ59h2AhHoJy5CVO5Oxp4JvDg4o",
  authDomain: "capstone-auth-f81a5.firebaseapp.com",
  projectId: "capstone-auth-f81a5",
  storageBucket: "capstone-auth-f81a5.firebasestorage.app",
  messagingSenderId: "306634879545",
  appId: "1:306634879545:web:5322e4941aa06660f013e8",
  measurementId: "G-C6KYH71Y9Q"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);

// Check if we are in the browser before initializing analytics
export const analytics = typeof window !== 'undefined' ? getAnalytics(app) : null;
