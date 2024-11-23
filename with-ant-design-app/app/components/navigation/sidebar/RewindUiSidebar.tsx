import Link from "next/link";
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
import { getGroupsForSidebar } from "../../../../actions/group";
import {
  usePathname,
  useSearchParams,
} from "next/navigation";
import { FaGears, FaPersonFalling } from "react-icons/fa6";

const RewindUiSidebar = () => {
  const [collapsed, setCollapsed] = useState(
    (typeof window !== "undefined" &&
    localStorage.getItem("sidebarCollapsed") === "true") || false);

  const toggleCollapsed = () => {
    setCollapsed(!collapsed);
    localStorage.setItem("sidebarCollapsed", (!collapsed).toString());
  }

  const pathname = usePathname();
  const params = useSearchParams();

  console.log("pathname", pathname, "params", params);

  const [groups, setGroups] = useState<any>(null);
  useEffect(() => {
    getGroupsForSidebar().then((groups) => {
      setGroups(groups);
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
        className={`${styles[`left-sidebar-collapse`]}
      ${
        collapsed
          ? styles[`left-sidebar-collapse-collapsed`]
          : ``
      }
      `}
        onClick={() => toggleCollapsed()}
      >
        <LeftSquareFilled />
      </div>
      <Sidebar
        width="320px"
        collapsed={collapsed}
        collapsedWidth="3.7rem"
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
            height: "4em",
            minWidth: "16em",
            padding: ".5em 1em",
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
            // padding: ".5em 0",
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
          className={styles["left-sidebar-company-logo"]}
        >
          <Logo />
          <span
            className={styles["left-sidebar-company-title"]}
          >
            Icn
          </span>
        </div>
        <Menu>
          <MenuItem
            icon={<HomeOutlined />}
            href="/"
            active={pathname === "/"}
          >
            Sākums
          </MenuItem>
          {/* </SubMenu> */}

          <SubMenu
            icon={<HddOutlined />}
            active={pathname === "/groups"}
            label={"Manas Grupas"}
          >
            {groups?.map((group: any) => (
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
              href="/groups/new"
            >
              Pievienot jaunu
            </MenuItem>
          </SubMenu>

          <SubMenu
            label="Mans profils"
            icon={<UserOutlined />}
            active={pathname === "/profile"}
          >
            <MenuItem
              icon={<FaPersonFalling />}
              active={
                pathname === "/profile" &&
                (params.get("page") === "0" ||
                  !params.get("page"))
              }
              href="/profile?page=0"
            >
              {/* <NavLink href="/profile?page=0"> */}
              Lietotāja informācija
              {/* </NavLink>   */}
            </MenuItem>
            <MenuItem
              icon={<HeartOutlined />}
              active={
                pathname === "/profile" &&
                params.get("page") === "1"
              }
              href="/profile?page=1"
            >
              {/* <NavLink href="/profile?page=1"> */}
              Atzīmetas dzīvesvietas
              {/* </NavLink> */}
            </MenuItem>
            <MenuItem
              active={
                pathname === "/profile" &&
                params.get("page") === "2"
              }
              icon={<FaGears />}
              href="/profile?page=2"
            >
              {/* <NavLink href="/profile?page=2"> */}
              Iestatījumi
              {/* </NavLink> */}
            </MenuItem>
          </SubMenu>
        </Menu>
      </Sidebar>
    </div>
  );
};

export default RewindUiSidebar;
