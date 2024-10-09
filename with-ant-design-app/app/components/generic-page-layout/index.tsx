"use client";

import React from 'react';
import { Breadcrumb, Layout, Menu, theme } from 'antd';
import Navbar from '../navigation/navbar';

import styles from './GenericPageLayout.module.css';

const { Header, Content, Footer } = Layout;

const items = new Array(15).fill(null).map((_, index) => ({
  key: index + 1,
  label: `nav ${index + 1}`,
}));

const GenericLayout: React.FC<{ children?: React.ReactNode, homePage?: boolean }> = ({ children, homePage }) => {
  const {
    token: { colorBgContainer, borderRadiusLG },
  } = theme.useToken();

  return (
    <>
    <Navbar toggle={()=>{}} homePage={homePage}/>
    <Layout>
      {/* <Header style={{ display: 'flex', alignItems: 'center', height: "5rem", padding: '0 48px' }}> */}
        
      {/* </Header> */}
      <Content style={{ margin: homePage ? 'none' : '6rem 48px', borderRadius: borderRadiusLG }}>
        {/* {!homePage && (
        <Breadcrumb style={{ margin: '16px 0' }}>
          <Breadcrumb.Item>Home</Breadcrumb.Item>
          <Breadcrumb.Item>List</Breadcrumb.Item>
          <Breadcrumb.Item>App</Breadcrumb.Item>
        </Breadcrumb>
        )} */}
        <div
          style={{
            background: colorBgContainer,
            minHeight: 280,
            // padding: 24,
            borderRadius: borderRadiusLG,
          }}
        >
          {children}
        </div>
      </Content>
      <Footer className={styles["footer-container"]}>
        <span className={styles["footer-title-span"]}>Cenu paredzēšana</span> ©{new Date().getFullYear()} autors Gustavs Jakobsons
      </Footer>
    </Layout>
    </>
  );
};

export default GenericLayout;