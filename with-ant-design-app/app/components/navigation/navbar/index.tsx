"use client";

import React, { useState } from "react";
import Link from "next/link";
import Logo from "./Logo";
import styles from "./Navbar.module.css"; // Import the new CSS module
import Column from "antd/es/table/Column";
import { Button, Divider } from "antd";
import { DownOutlined } from "@ant-design/icons";
import SignUpModal from "./sign-up-modal";
import LoginModal from "./log-in-modal";

const Navbar = ({ toggle, homePage }: { toggle: () => void, homePage: boolean | undefined }) => {

  const [signUpModalOpen, setSignUpModalOpen] = useState(false);
  const [loginModalOpen, setLoginModalOpen] = useState(false);

  return (
    <div className={`${styles.navbar} ${homePage ? "" : styles["navbar-with-background"]}`}>
      <div className={styles.container}>
        <div className={styles.content}>
          <div className={styles["logo-container"]}>
            <Logo />
          </div>
          <div className={styles["menu-container"]}>
            <Link href="/#about" className={styles["menu-item"]}>
              <span>Par lapu</span>
              <DownOutlined className={styles["down-icon"]} />
            </Link>
            <Link href="/#services" className={styles["menu-item"]}>
              <span>Mani dati</span>
            </Link>
            <Link href="/#projects" className={styles["menu-item"]}>
              <span>Projekti</span>
            </Link>

            <Link href="/#contacts" className={styles["menu-item"]}>
              <span>Kontakti</span>
            </Link>
          </div>

            <div className={styles["profile-container"]}>
            <Button className={styles["rounded-button"]}
              onClick={() => {
                setLoginModalOpen(true);
              }}
            >
              <span>Autorizēties</span>
            </Button>
            <LoginModal open={loginModalOpen} setOpen={setLoginModalOpen} />
            <Button className={`${styles["rounded-button"]} ${styles["button-fill"]}`}
              // ghost
              onClick={() =>{
                setSignUpModalOpen(true);
              }}>
              <span>Izveidot jaunu kontu</span>
            </Button>
            <SignUpModal open={signUpModalOpen} setOpen={setSignUpModalOpen} />
            </div>
        </div>
      </div>
    </div>
  );
};

export default Navbar;
