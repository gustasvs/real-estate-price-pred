// PageHeader component with TypeScript support for breadcrumbs and a title

// "use client";

import React from 'react';
import { Breadcrumb } from 'antd';
import { HomeOutlined } from '@ant-design/icons';

import styles from './PageHeader.module.css';
// import { useRouter } from 'next/navigation';

import Link from 'next/link';

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
    <div className={styles.pageHeader}>
      <Breadcrumb>
        <Breadcrumb.Item className={styles.breadcrumbItem}>
          <Link href="/" className={styles.link}> 
            <HomeOutlined />
          </Link>
        </Breadcrumb.Item>
        {breadcrumbItems.map((item, index) => (
          <Breadcrumb.Item key={index} className={styles.breadcrumbItem}>
            <Link href={item.path} className={styles.link}>
              {item.label}
            </Link>
          </Breadcrumb.Item>
        ))}
      </Breadcrumb>
      <span className={styles.title}>{title}</span>
    </div>
  );
};

export default PageHeader;