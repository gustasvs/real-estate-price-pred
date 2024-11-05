// PageHeader component with TypeScript support for breadcrumbs and a title

import React from 'react';
import { Breadcrumb } from 'antd';
import { HomeOutlined } from '@ant-design/icons';

import styles from './PageHeader.module.css';

type BreadcrumbItem = {
  label: string;
  path: string;
};

interface PageHeaderProps {
  title: string;
  breadcrumbItems: BreadcrumbItem[];
}

const PageHeader: React.FC<PageHeaderProps> = ({ title, breadcrumbItems }) => {
  return (
    <div>
      <Breadcrumb
        >
        <Breadcrumb.Item
          className={styles.breadcrumbItem}

        >
          <HomeOutlined />
        </Breadcrumb.Item>
        {breadcrumbItems.map((item, index) => (
          <Breadcrumb.Item href={item.path} key={index}
          className={styles.breadcrumbItem}
          >
            <span>{item.label}</span>
          </Breadcrumb.Item>
        ))}
      </Breadcrumb>
      <span className={styles.title}>{title}</span>
    </div>
  );
};

export default PageHeader;
