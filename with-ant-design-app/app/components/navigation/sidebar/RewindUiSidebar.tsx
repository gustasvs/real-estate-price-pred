import Link from "next/link";
import { useState } from "react";
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
import { LeftSquareFilled, LeftSquareOutlined } from "@ant-design/icons";

const RewindUiSidebar = () => {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <Sidebar
      collapsed={collapsed}
      rootStyles={{
        [`.${sidebarClasses.container}`]: {
            backgroundColor: 'var(--background-dark-main)',
          },
        [`.${menuClasses.button}`]: {
            color: 'var(--background-light-secondary)',
            backgroundColor: 'var(--background-dark-secondary)',

            '&:hover': {
                backgroundColor: 'var(--background-dark-secondary)',
                color: 'var(--background-light-main)',
            },
          },
        [`.${menuClasses.active}`]: {
                backgroundColor: 'var(--background-dark-main)',
                color: 'var(--text-brighter)',
            },

        [`.${menuClasses.subMenuContent}`]: {
            backgroundColor: 'var(--background-dark-secondary)',
            // borderRadius: '1em',
            },
            // hover
      }}
    >
      <div className={styles["left-sidebar-company-logo"]}>
        <Logo />
        <span
          className={styles["left-sidebar-company-title"]}
        >
          Icn
        </span>
        <div className={styles[`left-sidebar-collapse`]}>
            <div
                className={styles[`${collapsed ? "collapsed" : ""}`]}
                onClick={() => setCollapsed(!collapsed)}
            >
                <LeftSquareFilled />
            </div>
        </div>
      </div>
      <Menu
        // menuItemStyles={{
        //     button: ({ level, active, disabled }) => {
        //         return {
        //           backgroundColor: active ? '#eecef9' : undefined,
        //         };
        //     },
        //   }}
      >
        <SubMenu label="Manas Grupas">
        
          <MenuItem
            active={true}
          > Pie charts </MenuItem>
          <MenuItem> Line charts </MenuItem>
        </SubMenu>
        <MenuItem> Documentation </MenuItem>
        <MenuItem> Calendar </MenuItem>
      </Menu>
    </Sidebar>
  );
};

export default RewindUiSidebar;
