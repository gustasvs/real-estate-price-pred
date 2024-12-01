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

const roboto = Roboto({
  weight: "400",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Inovācija cenu noteikšanā",
  description: "Nosakiet cenu, izmantojot inovatīvus risinājumus",
};

function RootLayout({ children }: React.PropsWithChildren) {
  return (
    <html lang="en">
      <body className={roboto.className}>
        <SessionProvider>
          <ConfigProvider theme={theme}>
            <ThemeProvider>
              <Navbar homePage={false} />
              {children}
            </ThemeProvider>
          </ConfigProvider>
        </SessionProvider>
      </body>
    </html>
  );
}

export default RootLayout;
