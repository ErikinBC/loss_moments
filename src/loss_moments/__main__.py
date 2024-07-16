"""
Script to test that package compiled properly:

python3 -m loss_moments
"""

if __name__ == '__main__':
    from ._examples import simulation
    simulation()

    print('\n\nModule working as expected')