import type { ThemeConfig } from "antd";

const theme: ThemeConfig = {
  components: {
    Upload: {
      fontSize: 16,
      colorPrimary: "#35393b",
      colorTextSecondary: "#dddddd",
      borderRadiusLG: 16,
      fontFamily: "inherit"
      // width: 200,
    },
    Popover: {
      colorBgElevated: "var(--background-dark-secondary)",
      colorText: "var(--background-light-secondary)",
      fontFamily: "inherit",
    },
    Breadcrumb: {
      fontSize: 16,
      colorText: "var(--background-light-main)",
      colorTextSecondary: "var(--background-light-main)",
      separatorColor: "var(--background-light-main)",
      linkColor: "var(--background-light-main)",
      linkHoverColor: "var(--text-brighter)"
    },
    Button: {
      defaultBg: "var(--background-dark-secondary)",
      defaultColor: "var(--background-light-secondary)",
      defaultHoverColor: "var(--text-bright)",
      defaultHoverBg: "var(--background-dark-main)",

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
