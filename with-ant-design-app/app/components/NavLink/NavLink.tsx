import Link from 'next/link';
import { CSSProperties, FC } from 'react';

interface NavLinkProps {
    href: string;
    as?: string;
    passHref?: boolean;
    prefetch?: boolean;
    replace?: boolean;
    scroll?: boolean;
    shallow?: boolean;
    locale?: string | false;
    className?: string;
    style?: CSSProperties;
    children: React.ReactNode;
}

const NavLink: FC<NavLinkProps> = ({
    href,
    as,
    passHref,
    prefetch,
    replace,
    scroll,
    shallow,
    locale,
    className,
    style,
    children,
}) => {
    return (
        <Link
            href={href}
            as={as}
            passHref={passHref}
            prefetch={prefetch}
            replace={replace}
            scroll={scroll}
            shallow={shallow}
            locale={locale}
            style={{ color: "inherit", textDecoration: "none",
                cursor: "pointer", ...style }}
        >
            {children}
        </Link>
    );
};

export default NavLink;