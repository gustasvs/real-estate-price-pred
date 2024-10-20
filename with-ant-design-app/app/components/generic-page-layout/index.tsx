"use client";

import React from 'react';
import { Breadcrumb, Divider, Layout, Menu, theme } from 'antd';
import Navbar from '../navigation/navbar';

import styles from './GenericPageLayout.module.css';
import { FacebookOutlined, GithubOutlined, GitlabOutlined, InstagramOutlined, XOutlined, YoutubeOutlined } from '@ant-design/icons';
import { useSession } from 'next-auth/react';

const { Header, Content, Footer } = Layout;

const items = new Array(15).fill(null).map((_, index) => ({
  key: index + 1,
  label: `nav ${index + 1}`,
}));

const GenericLayout: React.FC<{ children?: React.ReactNode, homePage?: boolean }> = ({ children, homePage }) => {
  const {
    token: { colorBgContainer, borderRadiusLG },
  } = theme.useToken();

  
  const { data: session, status } = useSession();

  return (
    <>
    <Navbar toggle={()=>{}} homePage={homePage}/>
    <Layout
      className={styles["layout-container"]}
      style={{ backgroundColor: "var(--background-dark-secondary)" }}
    >
      <Content style={{ margin: homePage ? 'none' : '6rem 48px 0px 48px', borderRadius: borderRadiusLG }}
        className={styles["site-layout"]}

      >
        {/* {!homePage && (
        <Breadcrumb style={{ margin: '16px 0' }}>
          <Breadcrumb.Item>Home</Breadcrumb.Item>
          <Breadcrumb.Item>List</Breadcrumb.Item>
          <Breadcrumb.Item>App</Breadcrumb.Item>
        </Breadcrumb>
        )} */}
        <div
          className={styles["site-layout-content"]}
        >
          {children}
        </div>
      </Content>
      <Footer className={styles["footer-container"]}>

          <Divider style={{borderColor: 'white'}}>
            <div className={styles["footer-social"]}>
              <FacebookOutlined className={styles["footer-social-icon"]} />
              <InstagramOutlined className={styles["footer-social-icon"]} />
              <YoutubeOutlined className={styles["footer-social-icon"]} />
              <XOutlined className={styles["footer-social-icon"]} />
              <GithubOutlined className={styles["footer-social-icon"]} />
            </div>

          </Divider>
          <div className={styles["footer-logo"]}>
            <GitlabOutlined className={styles["footer-logo-icon"]} />
            <div className={styles["footer-logo-text"]}>
              <span className={styles["footer-logo-title"]}>
              "Inovācija cenu noteikšanā" 
              </span>
              <br></br> 
              Visas tiesības aizsargātas ©{new Date().getFullYear()}
            </div>
          </div>
          {/* <div className={styles["footer-links"]}>
  <a href="/legal-stuff">Legal Stuff</a>
  <a href="/privacy-policy">Privacy Policy</a>
  <a href="/security">Security</a>
  <a href="/website-accessibility">Website Accessibility</a>
  <a href="/manage-cookies">Manage Cookies</a>
</div> */}
<div className={styles["footer-links"]}>
  <a href="/legal-stuff">Juridiskā informācija</a>
  <a href="/privacy-policy">Privātuma politika</a>
  <a href="/security">Drošība</a>
  <a href="/website-accessibility">Vietnes pieejamība</a>
  <a href="/manage-cookies">Pārvaldīt sīkdatnes</a>
</div>


      
      </Footer>
    </Layout>
    </>
  );
};

export default GenericLayout;