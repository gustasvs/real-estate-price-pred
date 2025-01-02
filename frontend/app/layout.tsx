import React from "react";
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import StyledComponentsRegistry from "./AntdRegistry";
import "./globals.css";
import { SessionProvider } from "next-auth/react";
import { ConfigProvider } from "antd";
import theme from "./themeConfig";

import { Roboto } from "next/font/google";
import { ThemeProvider } from "./context/ThemeContext";
import Navbar from "./components/navigation/navbar";
import { Providers } from "./providers";

const roboto = Roboto({
  weight: "400",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "SmartEstate",
  description: "Nosakiet cenu, izmantojot inovatīvus risinājumus",
};

function RootLayout({ children }: React.PropsWithChildren) {
  return (
    <html lang="en">
      <body className={roboto.className}>
        
          <ConfigProvider theme={theme}>
          <Providers>
            <ThemeProvider>
              <Navbar homePage={false} />
              
                {children}
            </ThemeProvider>
            </Providers>
          </ConfigProvider>
      </body>
    </html>
  );
}

export default RootLayout;
