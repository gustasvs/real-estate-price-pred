// "use client";

import React from "react";
import {
  Breadcrumb,
  Divider,
  Layout,
  Menu,
  theme,
} from "antd";
import Navbar from "../navigation/navbar";

import styles from "./GenericPageLayout.module.css";
import RewindUiSidebar from "../navigation/sidebar/RewindUiSidebar";
import PageFooter from "./page-footer/PageFooter";

const { Header, Content, Footer } = Layout;

const GenericLayout: React.FC<{
  children?: React.ReactNode;
  homePage?: boolean;
}> = ({ children, homePage }) => {

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: "row",
        }}
      >
        {!homePage && <RewindUiSidebar />}
        <div
          className={styles["layout-container"]}
          style={{
            backgroundColor: "var(--background-dark-main)",
            width: "100%",
            minHeight: "100vh",
          }}
        >
          <div
            style={{
              margin: homePage
                ? "none"
                : "2rem 2rem 3rem 3rem",
              borderRadius: "10px",
            }}
            className={styles["site-layout"]}
          >
            <div className={styles["site-layout-content"]}>
              {children}
            </div>
          </div>
        </div>
      </div>

      <PageFooter />
    </div>
  );
};

export default GenericLayout;
