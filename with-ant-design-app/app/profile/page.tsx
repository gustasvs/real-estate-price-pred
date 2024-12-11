import styles from "./Profile.module.css";
import GenericLayout from "../components/generic-page-layout";
import MyProfileForm from "../components/my-profile/my-profile-form/MyProfileForm";

import PageHeader from "../components/generic-page-layout/page-header/PageHeader";

const UserProfilePage = async () => {

  return (
    <GenericLayout>
      <PageHeader
        title="Mans profils"
        breadcrumbItems={[
          {
            label: "Mans profils",
            path: "/profile",
          },
        ]}
      />

      <div className={styles["profile-page-container"]}>

          {/* <Sidebar sidebarItems={sidebarItems} activeNavItem={activeNavItem} onNavClick={handleNavClick} title="Mans profils" /> */}

        <div className={`${styles['main-content']}`}>
          <MyProfileForm />
        </div>
      </div>
    </GenericLayout>
  );
};

export default UserProfilePage;
