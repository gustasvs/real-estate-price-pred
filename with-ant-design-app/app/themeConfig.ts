import type { ThemeConfig } from "antd";

const theme: ThemeConfig = {
  components: {
    Upload: {
      fontSize: 16,
      colorPrimary: "#35393b",
      colorTextSecondary: "#dddddd",
      borderRadiusLG: 16,
      fontFamily: "inherit",
      colorFillAlter: "var(--background-dark-main-hover)",
      // width: 200,
    },
    Popover: {
      colorBgElevated: "var(--background-dark-secondary)",
      colorText: "var(--background-light-secondary)",
      fontFamily: "inherit",
    },
    Breadcrumb: {
      // fontSize: 16,
      colorText: "var(--background-light-main)",
      colorTextSecondary: "var(--background-light-main)",
      separatorColor: "var(--background-light-main)",
      linkColor: "var(--background-light-main)",
      linkHoverColor: "var(--text-brighter)"
    },
    Button: {
      defaultBg: "transparent", // default button background
      defaultColor: "var(--background-light-main)",
      defaultHoverColor: "var(--text-bright)",
      defaultHoverBg: "var(--background-dark-secondary)", // default button background on hover
      defaultBorderColor: "var(--background-light-main)", // default button border color

      dangerColor: "var(--background-light-main)", // danger button text color

      defaultActiveBg: "var(--background-dark-main)",
      defaultActiveColor: "var(--text-brighter)",

      colorPrimary: "var(--background-light-main)", // primary button background
      primaryColor: "var(--background-dark-main)", // primary button text color on hover..
      colorPrimaryHover: "var(--background-light-secondary)", // primary button background on hover

      colorPrimaryActive: "var(--background-light-secondary)", // primary button background on active
      colorPrimaryTextActive: "var(--background-dark-main)", // primary button text color on active

    },
    Modal: {
      contentBg: "var(--background-dark-main)",
      headerBg: "var(--background-dark-main)",
      colorTextHeading: "var(--background-light-secondary)",
      colorText: "var(--background-light-secondary)",
    },
    Divider: {
      colorText: "var(--background-light-main)",
      colorBorder: "var(--background-light-main)",
    },
    Popconfirm: {
      colorText: "var(--background-light-secondary)",
      colorTextHeading: "var(--background-light-secondary)",
      colorTextDescription: "var(--background-light-secondary)",
      colorBgElevated: "var(--background-dark-secondary)",
    },
    Pagination: {
      colorBgContainer: "var(--background-dark-main-hover)",
      itemActiveBg: "var(--text-bright)",
      controlItemBgActiveHover: "var(--text-brighter)",
      // itemActiveBgDisabled: "rgba(0, 0, 0, 0.15)",
      // itemActiveColorDisabled: "rgba(255, 255, 255, 0.25)",
      itemBg: "var(--background-light-main)",
      controlItemBgHover: "var(--text-bright)",
      itemInputBg: "var(--background-light-main)",
      itemLinkBg: "var(--background-light-main)",
      colorIcon: "var(--background-light-secondary)",
      colorIconHover: "var(--text-bright)",
      colorBgTextHover: "var(--text-bright)",
      colorText: "var(--background-darkest)",
    },
    Select: {
      colorText: "var(--background-light-secondary)",
      colorTextPlaceholder: "var(--background-light-secondary)",
      colorTextDisabled: "var(--background-light-secondary)",
      colorBorder: "var(--background-light-main)",
      colorBgBase: "var(--background-dark-main-hover)",
    }
    
  },
  token: {
    fontSize: 16,
    colorPrimary: "#35393b",
    colorTextSecondary: "#dddddd",
    borderRadiusLG: 16,
    fontFamily: "inherit"
  },
};

export default theme;
