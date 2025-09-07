import { useContext, useEffect, useState } from "react";
import {auth} from "../../firebase"
import { onAuthStateChanged } from "firebase/auth";
import { initialize } from "next/dist/server/lib/render-server";

const Authcontext = React.createContext();

export function useAuth(){
  return useContext(AuthContext);
}

export function AuthProvider({children}) {
  const [currentUser, setCurrentUser] = useState(null);
  const [currentLoggedIn, setLoggedIn] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(()=>{
        const unsubscribe = onAuthStateChanged(auth, initializeUser);

  }, [])

  async function initializeUser(user) {
    if (user) {
        setCurrentUser({...user});
        setUserLoggedIn(true);
    } else {
        setCurrentUser(null);
        setLoggedIn(false);
    }
    setLoading(false);
    }

    const value = {
      currentUser,
      userLoggedIn,
      loading
    }

    return(
        <AuthContext.Provider value={value}>
            {!loading && children}
          </AuthContext.Provider>

    )
  }