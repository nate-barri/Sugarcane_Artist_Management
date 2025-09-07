// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getAuth } from "firebase/auth"; // You need to import this here

// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyCpdSrT2I45vy3dgv9pqlfSCLLS_57iuPI",
  authDomain: "sugar-cane-935a7.firebaseapp.com",
  databaseURL: "https://sugar-cane-935a7-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "sugar-cane-935a7",
  storageBucket: "sugar-cane-935a7.firebasestorage.app",
  messagingSenderId: "165955215189",
  appId: "1:165955215189:web:11a25a617c03bb419199dc",
  measurementId: "G-L6TM5BYL7G"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app) // Corrected: getAuth with a capital A
const analytics = getAnalytics(app);

export {app, auth};