// context/ThemeContext.tsx
'use client';

import { createContext, useContext, ReactNode, useState, useEffect } from 'react';
import { useSession } from 'next-auth/react';
import { updateUserTheme } from '../../actions/user';

interface ThemeContextType {
  theme: string;
  toggleTheme: () => void;

  fontSize: number;
  setFontSize: (size: number) => void;

  sidebarPreferedOpen: boolean;
  toggleSidebarPreferedOpen: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

interface ThemeProviderProps {
  children: ReactNode;
}

export function ThemeProvider({ children }: ThemeProviderProps) {
  const { data: session } = useSession();

  // Initialize local theme state based on session data or default to 'light'
  const [theme, setTheme] = useState(session?.user?.theme || 'light');

  useEffect(() => {
    // Sync local theme state with session data if it changes
    if (session?.user?.theme && session.user.theme !== theme) {
      setTheme(session.user.theme);
    }
  }, [session?.user?.theme]);

  useEffect(() => {
    // Update <html> class based on theme
    if (theme === 'light') {
      document.documentElement.classList.add('light-theme');
    } else {
      document.documentElement.classList.remove('light-theme');
    }
  }, [theme]);

  const toggleTheme = async (newTheme: string) => {

    console.log("called theme swithc to", newTheme);
    setTheme(newTheme);

    updateUserTheme(newTheme);

    // Update theme in session user object and database

    if (session?.user) {
      session.user.theme = newTheme;
    }
  };

  const fontSize = 16; // Default font size
  const setFontSize = (size: number) => {
    // Update font size logic
  };

  const sidebarPreferedOpen = true; // Default sidebar state
  const toggleSidebarPreferedOpen = () => {
    // Toggle sidebar state logic
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme, fontSize, setFontSize, sidebarPreferedOpen, toggleSidebarPreferedOpen }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useThemeContext() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
