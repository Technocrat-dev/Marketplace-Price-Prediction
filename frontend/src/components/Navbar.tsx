'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useSession, signIn, signOut } from 'next-auth/react';
import styles from './Navbar.module.css';

const navLinks = [
    { href: '/', label: 'Predict' },
    { href: '/dashboard', label: 'Dashboard' },
    { href: '/explore', label: 'Explore' },
    { href: '/about', label: 'About' },
];

export default function Navbar() {
    const pathname = usePathname();
    const [mobileOpen, setMobileOpen] = useState(false);
    const { data: session, status } = useSession();

    return (
        <nav className={styles.nav} aria-label="Main navigation">
            <div className={styles.inner}>
                <Link href="/" className={styles.logo} aria-label="PriceScope home">
                    <span className={styles.logoIcon}>₹</span>
                    PriceScope
                </Link>

                {session && (
                    <>
                        <button
                            className={styles.hamburger}
                            onClick={() => setMobileOpen(!mobileOpen)}
                            aria-label="Toggle navigation menu"
                            aria-expanded={mobileOpen}
                        >
                            <span />
                            <span />
                            <span />
                        </button>

                        <div className={`${styles.links} ${mobileOpen ? styles.linksOpen : ''}`}>
                            {navLinks.map(({ href, label }) => (
                                <Link
                                    key={href}
                                    href={href}
                                    className={`${styles.link} ${pathname === href ? styles.linkActive : ''}`}
                                    onClick={() => setMobileOpen(false)}
                                >
                                    {label}
                                </Link>
                            ))}
                        </div>
                    </>
                )}

                {session?.user && (
                    <div className={styles.authSection}>
                        <div className={styles.userMenu}>
                            {session.user.image && (
                                <img
                                    src={session.user.image}
                                    alt={session.user.name || 'User'}
                                    className={styles.avatar}
                                    referrerPolicy="no-referrer"
                                />
                            )}
                            <span className={styles.userName}>{session.user.name?.split(' ')[0]}</span>
                            <button
                                onClick={() => signOut({ callbackUrl: '/login' })}
                                className={styles.signOutBtn}
                                aria-label="Sign out"
                            >
                                Sign Out
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </nav>
    );
}
