"use client";

import React, { useState } from "react";
import Link from "next/link";
import Logo from "./Logo";
import styles from "./Navbar.module.css"; // Import the new CSS module
import Column from "antd/es/table/Column";
import { Button, Divider } from "antd";
import { DownOutlined } from "@ant-design/icons";
import SignUpModal from "./login-modal";

const Navbar = ({ toggle }: { toggle: () => void }) => {

  const [signUpModalOpen, setSignUpModalOpen] = useState(false);

  return (
    <div className={styles.navbar}>
      <div className={styles.container}>
        <div className={styles.content}>
          <div className={styles["logo-container"]}>
            <Logo />
          </div>
          <div className={styles["menu-container"]}>
            <Link href="/#about" className={styles["menu-item"]}>
              <span>About</span>
              <DownOutlined className={styles["down-icon"]} />
            </Link>
            <Link href="/#services" className={styles["menu-item"]}>
              <span>Services</span>
            </Link>
            <Link href="/#contacts" className={styles["menu-item"]}>
              <span>Contacts</span>
            </Link>
          </div>

            <div className={styles["profile-container"]}>
            <Button className={styles["rounded-button"]}>
              <span>Ieiet</span>
            </Button>
            <Button className={`${styles["rounded-button"]} ${styles["button-fill"]}`}
              onClick={() =>{
                setSignUpModalOpen(true);
              }}>
              <span>Izveidot kontu</span>
              <SignUpModal open={signUpModalOpen} setOpen={setSignUpModalOpen} />
            
            </Button>
            </div>
        </div>
      </div>
    </div>
  );
};

export default Navbar;
