"use client";

import { useEffect, useState } from "react";
import {
  Sidebar,
  Menu,
  MenuItem,
  SubMenu,
  sidebarClasses,
  menuClasses,
} from "react-pro-sidebar";
import Logo from "../navbar/Logo";

import styles from "./Sidebar.module.css";
import {
  HddOutlined,
  HeartOutlined,
  HomeOutlined,
  LeftSquareFilled,
  LeftSquareOutlined,
  PlusOutlined,
  UserOutlined,
} from "@ant-design/icons";

import { MdOutlineKeyboardDoubleArrowLeft } from "react-icons/md";

import { getGroupsForSidebar } from "../../../../actions/group";
import {
  usePathname,
  useRouter,
  useSearchParams,
} from "next/navigation";
import { FaGears, FaPersonFalling } from "react-icons/fa6";
import Link from "next/link";
import { BiSolidBuildings } from "react-icons/bi";
import { useThemeContext } from "../../../context/ThemeContext";

const RewindUiSidebar = () => {

  const router = useRouter();
  
  const { theme } = useThemeContext();

  const [collapsed, setCollapsed] = useState(
    (typeof window !== "undefined" &&
    localStorage.getItem("sidebarCollapsed") === "true") || false);

  const toggleCollapsed = () => {
    setCollapsed(!collapsed);
    localStorage.setItem("sidebarCollapsed", (!collapsed).toString());
  }

  const pathname = usePathname();
  const params = useSearchParams();

  // console.log("pathname", pathname, "params", params);

  const [groups, setGroups] = useState<any>(null);
  useEffect(() => {
    getGroupsForSidebar().then((groups) => {
      // setGroups(groups);
      setGroups(Array.isArray(groups) ? groups : []);
    });
  }, []);

  return (
    <div
      style={{
        display: "flex",
        position: "relative",
      }}
    >
      <div
        className={`${styles[`left-sidebar-collapse`]} ${Boolean(collapsed) && styles[`left-sidebar-collapse-collapsed`]}`}
        onClick={() => toggleCollapsed()}
      >
        <MdOutlineKeyboardDoubleArrowLeft />
      </div>
      <Sidebar
        width="320px"
        collapsed={collapsed}
        collapsedWidth="3.74rem"
        rootStyles={{
          [`.${menuClasses.subMenuRoot}`]: {
            borderRadius: "10px",
          },
          [`.${sidebarClasses.container}`]: {
            color: "var(--background-light-main)",
            backgroundColor: "var(--background-dark-main)",
            paddingLeft: ".5rem",
            fontSize: ".8rem",
          },
          [`.${menuClasses.button}`]: {
            borderRadius: "10px",
            color: "var(--background-light-main)",
            backgroundColor: "var(--background-dark-main)",
            height: "3.5em",
            minWidth: collapsed ? "1em" : "16em",
            padding: ".5em 1em",
            // margin: ".5em 0",
            transition: collapsed ? "min-width 0.1s 0.2s" : "min-width 0.1s",
            a: {
              textWrap: "wrap",
            },
            "&:hover": {
              borderRadius: "10px",
              backgroundColor:
                "var(--background-dark-secondary) !important",
              color: "var(--background-light-secondary)",
            },
          },
          [`.${sidebarClasses.root}.${sidebarClasses.collapsed}`]: {
            [`.${menuClasses.button}`]: {
              minWidth: "1em",
            },
          },
          [`.${menuClasses.button}.${menuClasses.active}.${menuClasses.open}`]: {
            backgroundColor: "var(--background-gray)",
            color: "var(--background-dark-main)",
            "&:hover": {
              backgroundColor: "var(--background-gray)",
              color: "var(--background-dark-main)",
            }
          },
          [`.${menuClasses.icon}`]: {
            svg: {
              fontSize: "0.5rem", // TODO not applied
            },
          },
          [`.${menuClasses.label}`]: {
            color: "inherit",
            backgroundColor: "inherit",
            textWrap: "wrap",
            textOverflow: "ellipsis",
            whiteSpace: "normal",
            overflow: "hidden",
            WebkitBoxOrient: "vertical",
            WebkitLineClamp: "2",
            display: "-webkit-box",
            lineHeight: "1.4em",
            "&:hover": {
              color: "inherit",
              backgroundColor: "inherit",
            },
          },
          [`.${menuClasses.icon}`]: {
            color: "inherit",
            backgroundColor: "inherit",
            "&:hover": {
              color: "inherit",
              backgroundColor: "inherit",
            },
          },
          [`.${menuClasses.menuItemRoot}`]: {
            borderRadius: "10px",
            // height: "3em",
            // padding: "1em 0",
          },
          [`.${menuClasses.subMenuContent}.${menuClasses.open}`]: {
            "&:before": {
              visibility: "visible",
            },
          },
          [`.${menuClasses.subMenuContent}`]: {
            // "&:before": {
            //   content: '""',
            //   position: "absolute",
            //   width: "2px",
            //   backgroundColor: "var(--background-light-main)",
            //   height: "60%",
            //   visibility: "hidden",
            //   left: "2rem",
            // },
            borderBottomRightRadius: "10px",
            marginLeft: "2rem",
            
            backgroundColor: "var(--background-dark-main)",
            [`.${menuClasses.menuItemRoot}`]: {
              width: "calc(100% - 2rem)",
              // "&:before": {
              //   content: '""',
              //   position: "absolute",
              //   border: "2px solid var(--background-light-main)",
              //   borderTop: "none",
              //   borderRight: "none",
              //   width: "1rem",
              //   height: "1rem",
              //   top: "50%",
              //   transform: "translateY(-75%)",
              //   borderBottomLeftRadius: "10px",
              // },
              "&:before": {
                // simple dot
                content: '""',
                position: "absolute",
                width: "5px",
                height: "5px",
                borderRadius: "50%",
                backgroundColor: "var(--background-light-main)",
                opacity: 0.7,
                top: "50%",
                // left: "rem",
                transform: "translateY(-50%)",
              }
            },
          },

          [`.${menuClasses.SubMenuExpandIcon}`]: {
            paddingBottom: "3px",
            scale: "110%",
            color: "var(--background-light-main)",
            // paddingRight: "1em",
          },
          [`.${menuClasses.subMenuContent} .${menuClasses.button}`]: {
            marginLeft: "1rem",
          },

            [`.${menuClasses.subMenuContent} .${menuClasses.button}.${menuClasses.active}`]: {
              backgroundColor: "var(--background-gray)",
              color: "var(--background-dark-main)",
              "&:hover": {
                backgroundColor:
                  "var(--background-light-secondary) !important",
                color: "var(--background-dark-main)",
              },
            },
        }}
      >
        <div
          className={`${styles["left-sidebar-company-logo"]} ${collapsed ? styles["left-sidebar-company-logo-collapsed"] : ""}`}
          onClick={() => router.push("/")}
          style={{
            filter: theme === "dark" ? "invert(1)" : "brightness(0) saturate(100%) invert(15%) sepia(9%) saturate(702%) hue-rotate(155deg) brightness(96%) contrast(90%)",
          }}
        >
          <Logo />
          {/* <span
            className={styles["left-sidebar-company-title"]}
          >
            Icn
          </span> */}
        </div>
        <Menu>
          <MenuItem
            icon={<HomeOutlined />}
            href="/"
            active={pathname === "/"}
          >
            {/* <Link href="/"> */}
              Sākums
            {/* </Link> */}
          </MenuItem>
          {/* </SubMenu> */}

          <SubMenu
            icon={<BiSolidBuildings />}
            active={pathname === "/groups"}
            label={"Manas Grupas"}
          >
            {groups && groups.length !== 0 && groups?.map((group: any) => (
              <MenuItem
                key={group.id}
                href={`/groups/${group.id}`}
                active={pathname === `/groups/${group.id}`}
                title={group.name}
              >
                {group.name}
              </MenuItem>
            ))}
            <MenuItem
              icon={<PlusOutlined />}
              href="/groups?new=true"
            >
              Pievienot jaunu
            </MenuItem>
          </SubMenu>

          {/* <SubMenu
            label="Mans profils"
            icon={<UserOutlined />}
            active={pathname === "/profile"}
          > */}
            <MenuItem
              icon={<UserOutlined />}
              active={pathname === "/profile"}
              href="/profile"
            >
              Mans profils
            </MenuItem>
            <MenuItem
              icon={<HeartOutlined />}
              active={pathname === "/profile/favourites"}
              href="/profile/favourites"
            >
              Atzīmetas dzīvesvietas
            </MenuItem>
            {/* <MenuItem
              active={
                pathname === "/profile" &&
                params.get("page") === "2"
              }
              icon={<FaGears />}
              href="/profile/favourites"
            >
              Iestatījumi
            </MenuItem> */}
          {/* </SubMenu> */}
        </Menu>
      </Sidebar>
    </div>
  );
};

export default RewindUiSidebar;
