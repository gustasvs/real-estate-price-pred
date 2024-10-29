"use client";

import React, { useState } from "react";
import Link from "next/link";
import Logo from "./Logo";
import styles from "./Navbar.module.css"; // Import the new CSS module
import Column from "antd/es/table/Column";
import { Button, Divider } from "antd";
import { DownOutlined, MoonOutlined, SunOutlined } from "@ant-design/icons";
import SignUpModal from "./sign-up-modal";
import LoginModal from "./log-in-modal";
import { getSession, useSession } from "next-auth/react";
import { Session } from "next-auth";
import { logout } from "../../../../actions/auth";
import Dropdown from "antd/es/dropdown/dropdown";
import { useRouter } from "next/navigation";
import UserIcon from "../../user-icon/UserIcon";

const Navbar = ({
  toggle,
  homePage,
}: {
  toggle: () => void;
  homePage: boolean | undefined;
}) => {
  const { data: session, status } = useSession() as {
    data: Session | null;
    status: string;
  };

  const router = useRouter();

  const [signUpModalOpen, setSignUpModalOpen] = useState(false);
  const [loginModalOpen, setLoginModalOpen] = useState(false);

  return (
    <div
      className={`${styles.navbar} ${
        homePage ? "" : styles["navbar-with-background"]
      }`}
    >
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
          {!session ? (
            <div className={styles["profile-container"]}>
              <Button
                className={styles["rounded-button"]}
                onClick={() => {
                  setLoginModalOpen(true);
                }}
              >
                <span>Autorizēties</span>
              </Button>
              <LoginModal open={loginModalOpen} setOpen={setLoginModalOpen} />
              <Button
                className={`${styles["rounded-button"]} ${styles["button-fill"]}`}
                // ghost
                onClick={() => {
                  setSignUpModalOpen(true);
                }}
              >
                <span>Izveidot jaunu kontu</span>
              </Button>
              <SignUpModal
                open={signUpModalOpen}
                setOpen={setSignUpModalOpen}
              />
            </div>
          ) : (
            <div className={styles["profile-container"]}>
              <Dropdown
              open={true}
                dropdownRender={(menu) => (
                  <div className={styles["profile-dropdown-container"]} >
                    <div className={styles["profile-dropdown-header"]}>
                      <UserIcon />
                      <div className={styles["profile-dropdown-header-text"]}>
                      <span className={styles["user-name"]}>
                          {session?.user?.name}
                        </span>
                        <span className={styles["user-email"]}>
                          {session?.user?.email}
                        </span>
                    </div>
                    </div>

                    <Divider className={styles["divider"]} />

                    <div className={styles["color-mode-container"]}>
                      <Button className={styles["color-mode-button"]}>
                        <SunOutlined />
                      </Button>
                      <Button className={styles["color-mode-button"]}>
                        <MoonOutlined />
                      </Button>
                    </div>
                    
                    <Button
                      className={styles["rounded-button"]}
                      onClick={() => {
                        router.push("/profile");
                      }}
                    >
                      <span>Mans profils</span>
                    </Button>

                    <Button
                      className={styles["rounded-button"]}
                      onClick={() => {
                        router.push("/groups");
                      }}
                    >
                      <span>Mani objekti</span>
                    </Button>

                    <Button
                      className={styles["rounded-button"]}
                      onClick={() => {
                        console.log("settings");
                      }}
                    >
                      <span>Iestatījumi</span>
                    </Button>

                    <Divider style={{ margin: "-0.5em 0", backgroundColor: "white"

                     }} />

                    <Button
                      className={styles["rounded-button"]}
                      onClick={async () => {
                        console.log("logging out");
                        await logout();
                        session ? console.log("logged out") : console.log("not logged out");
                      }}
                    >
                      <span>Izrakstīties</span>
                    </Button>
                  </div>
                )}
              >
                <div className={styles["user-box"]}>
                <UserIcon />
                <span className={styles["user-email"]}>
                  {session?.user?.name || session?.user?.email}
                </span>
                <DownOutlined className={styles["down-icon"]} />
                </div>
              </Dropdown>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Navbar;
