"use client";

import React, { useState } from "react";
import Link from "next/link";
import Logo from "./Logo";
import styles from "./Navbar.module.css"; // Import the new CSS module
import Column from "antd/es/table/Column";
import { Button, Divider } from "antd";
import {
  DownOutlined,
  HddOutlined,
  HeartFilled,
  HeartOutlined,
  MoonFilled,
  MoonOutlined,
  SunFilled,
  SunOutlined,
  UserOutlined,
} from "@ant-design/icons";
import SignUpModal from "./sign-up-modal";
import LoginModal from "./log-in-modal";
import { getSession, useSession } from "next-auth/react";
import { Session } from "next-auth";
import { logout } from "../../../../actions/auth";
import Dropdown from "antd/es/dropdown/dropdown";
import { useRouter } from "next/navigation";
import UserIcon from "../../user-icon/UserIcon";
import ColorThemeSwitch from "./dark-mode-switch/ColorThemeSwitch";
import {
  IoLogOutOutline,
  IoLogOutSharp,
  IoSettings,
  IoSettingsOutline,
  IoSettingsSharp,
} from "react-icons/io5";
import {
  FaHouse,
  FaHouseChimneyWindow,
} from "react-icons/fa6";

import { BiDotsVertical, BiLogOut } from "react-icons/bi";
import { FaHamburger } from "react-icons/fa";

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

  const [currentTheme, setCurrentTheme] = useState("light");

  const [signUpModalOpen, setSignUpModalOpen] =
    useState(false);
  const [loginModalOpen, setLoginModalOpen] =
    useState(false);

  return (
    <div
      className={`${styles.navbar} ${
        homePage ? "" : styles["navbar-with-background"]
      }`}
    >
      <div className={styles.container}>
        <div className={styles.content}>
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
              <LoginModal
                open={loginModalOpen}
                setOpen={setLoginModalOpen}
              />
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
                // open={true}
                dropdownRender={(menu) => (
                  <div
                    className={
                      styles["profile-dropdown-container"]
                    }
                  >
                    <div
                      className={
                        styles["profile-dropdown-header"]
                      }
                    >
                      <UserIcon />
                      <div
                        className={
                          styles[
                            "profile-dropdown-header-text"
                          ]
                        }
                      >
                        <span
                          className={styles["user-name"]}
                        >
                          {session?.user?.name}
                        </span>
                        <span
                          className={styles["user-email"]}
                        >
                          {session?.user?.email}
                        </span>
                      </div>
                    </div>

                    <Divider
                      className={styles["divider"]}
                    />

                    <ColorThemeSwitch
                      currentTheme={currentTheme}
                      setCurrentTheme={setCurrentTheme}
                    />

                    <Divider
                      className={styles["divider"]}
                    />

                    <div
                      className={
                        styles[
                          "profile-dropdown-navigation-button"
                        ]
                      }
                      onClick={() => {
                        router.push("/profile");
                      }}
                    >
                      <UserOutlined />
                      <span>Mans profils</span>
                    </div>

                    <div
                      className={
                        styles[
                          "profile-dropdown-navigation-button"
                        ]
                      }
                      onClick={() => {
                        router.push("/groups");
                      }}
                    >
                      <FaHouse />
                      <span>Mani objekti</span>
                    </div>

                    <div
                      className={
                        styles[
                          "profile-dropdown-navigation-button"
                        ]
                      }
                      onClick={() => {
                        router.push("/profile?page=1");
                      }}
                    >
                      <HeartFilled />
                      <span>Mani atzīmētie objekti</span>
                    </div>

                    <div
                      className={
                        styles[
                          "profile-dropdown-navigation-button"
                        ]
                      }
                      onClick={() => {
                        router.push("/profile?page=2");
                      }}
                    >
                      <IoSettingsSharp />
                      <span>Iestatījumi</span>
                    </div>

                    <Divider
                      className={styles["divider"]}
                    />

                    <div
                      className={
                        styles[
                          "profile-dropdown-navigation-button"
                        ]
                      }
                      onClick={() => {
                        console.log("logout");
                        logout()
                      }}
                    >
                      <BiLogOut />
                      <span>Iziet</span>
                    </div>
                  </div>
                )}
              >
                <div className={styles["user-box"]}>
                  <UserIcon />
                  <span className={styles["user-email"]}>
                    {session?.user?.name ||
                      session?.user?.email}
                  </span>
                  <BiDotsVertical />
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
