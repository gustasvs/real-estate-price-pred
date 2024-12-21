// context/ThemeContext.tsx
"use client";

import {
  createContext,
  useContext,
  ReactNode,
  useState,
  useEffect,
} from "react";
import { useSession } from "next-auth/react";
import { updateUserTheme } from "../../actions/user";

interface ThemeContextType {
  theme: string;
  toggleTheme: (newTheme: string) => void;

  fontSize: number;
  setFontSize: (size: number) => void;

  sidebarPreferedOpen: boolean;
  toggleSidebarPreferedOpen: () => void;
}

const ThemeContext = createContext<
  ThemeContextType | undefined
>(undefined);

interface ThemeProviderProps {
  children: ReactNode;
}

export function ThemeProvider({
  children,
}: ThemeProviderProps) {
  const { data: session } = useSession();

  // Initialize local theme state based on session data or default to 'light'
  const [theme, setTheme] = useState(
    session?.user?.theme ||
    (typeof window !== "undefined" && localStorage.getItem("theme")) || // Add check for window
    "light"
  );

  const [fontSize, setFontSizeState] = useState<number>(
    parseInt(
      (typeof window !== "undefined" && localStorage.getItem("fontSize")) || session?.user?.fontSize?.toString() || "19"
    )
  );

  useEffect(() => {
    if (
      session?.user?.theme &&
      session.user.theme !== theme
    ) {
      setTheme(session.user.theme);
    }
  }, [session?.user?.theme]);

  useEffect(() => {
    if (theme === "light") {
      document.documentElement.classList.add("light-theme");
    } else {
      document.documentElement.classList.remove(
        "light-theme"
      );
    }
    localStorage.setItem("theme", theme);
  }, [theme]);


    // Update font size in root element and localStorage
    useEffect(() => {

      if (typeof fontSize !== "number") return;

      if (fontSize < 12 || fontSize > 26) return;

      document.documentElement.style.setProperty(
        "font-size",
        `${fontSize}px`
      );
      localStorage.setItem("fontSize", fontSize.toString());
    }, [fontSize]);

    const setFontSize = (size: number) => {

      if (size < 12 || size > 26) return;
      setFontSizeState(size);

      if (session?.user) {
        session.user.fontSize = size.toString();
      }
    };


  const toggleTheme = async (newTheme: string) => {
    console.log("called theme swithc to", newTheme);
    setTheme(newTheme);

    updateUserTheme(newTheme);

    // Update theme in session user object and database

    if (session?.user) {
      session.user.theme = newTheme;
    }
  };


  const sidebarPreferedOpen = true; // Default sidebar state
  const toggleSidebarPreferedOpen = () => {
    // Toggle sidebar state logic
  };

  return (
    <ThemeContext.Provider
      value={{
        theme,
        toggleTheme,
        fontSize,
        setFontSize,
        sidebarPreferedOpen,
        toggleSidebarPreferedOpen,
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
}

export function useThemeContext() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error(
      "useTheme must be used within a ThemeProvider"
    );
  }
  return context;
}
