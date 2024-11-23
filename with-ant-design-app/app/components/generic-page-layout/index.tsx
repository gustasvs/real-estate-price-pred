"use client";

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
import {
  FacebookOutlined,
  GithubOutlined,
  GitlabOutlined,
  InstagramOutlined,
  LeftSquareFilled,
  XOutlined,
  YoutubeOutlined,
} from "@ant-design/icons";
import { useSession } from "next-auth/react";
import Sidebar from "../navigation/sidebar/Sidebar";
import RewindUiSidebar from "../navigation/sidebar/RewindUiSidebar";

const { Header, Content, Footer } = Layout;

const GenericLayout: React.FC<{
  children?: React.ReactNode;
  homePage?: boolean;
}> = ({ children, homePage }) => {
  
  const {
    token: { colorBgContainer, borderRadiusLG },
  } = theme.useToken();

  const { data: session, status } = useSession();

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
        <Navbar toggle={() => {}} homePage={homePage} />
        {!homePage && (
          <RewindUiSidebar />
        )}
        <Layout
          className={styles["layout-container"]}
          style={{
            backgroundColor: "var(--background-dark-main)",
            width: "100%",
            height: "100%",
          }}
        >
          <Content
            style={{
              margin: homePage
                ? "none"
                : "2rem 2rem 3rem 3rem",
              borderRadius: borderRadiusLG,
            }}
            className={styles["site-layout"]}
          >
            {/* {!homePage && (
        <Breadcrumb style={{ margin: '16px 0' }}>
          <Breadcrumb.Item>Home</Breadcrumb.Item>
          <Breadcrumb.Item>List</Breadcrumb.Item>
          <Breadcrumb.Item>App</Breadcrumb.Item>
        </Breadcrumb>
        )} */}
            <div className={styles["site-layout-content"]}>
              {children}
            </div>
          </Content>
        </Layout>
      </div>

      <Footer className={styles["footer-container"]}>
        <Divider style={{ borderColor: "white" }}>
          <div className={styles["footer-social"]}>
            {/* <FacebookOutlined
              className={styles["footer-social-icon"]}
            /> */}
            {/* <InstagramOutlined
              className={styles["footer-social-icon"]}
            /> */}
            <YoutubeOutlined
              className={styles["footer-social-icon"]}
            />
            <XOutlined
              className={styles["footer-social-icon"]}
            />
            <GithubOutlined
              className={styles["footer-social-icon"]}
            />
          </div>
        </Divider>
        <div className={styles["footer-logo"]}>
          <GitlabOutlined
            className={styles["footer-logo-icon"]}
          />
          <div className={styles["footer-logo-text"]}>
            <span className={styles["footer-logo-title"]}>
              "Inovācija cenu noteikšanā"
            </span>
            <br></br>
            Visas tiesības aizsargātas ©
            {new Date().getFullYear()}
          </div>
        </div>
        <div className={styles["footer-links"]}>
          <a href="/legal-stuff">Juridiskā informācija</a>
          <a href="/privacy-policy">Privātuma politika</a>
          <a href="/security">Drošība</a>
          <a href="/website-accessibility">
            Vietnes pieejamība
          </a>
          <a href="/manage-cookies">Pārvaldīt sīkdatnes</a>
        </div>
      </Footer>
    </div>
  );
};

export default GenericLayout;
